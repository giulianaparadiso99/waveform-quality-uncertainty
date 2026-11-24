"""Building blocks shared by the experiments that train several models (learning curve, search).

An experiment trains models on one or more training/validation partitions of the records.
For each partition the normalization statistics and the bad-class weight come from its
training records only; the statistics are cached in a file, so that later sessions and
other models on the same partition reuse them.

Cross-validation partitions (``fold_partitions``) use the folds of ``assign_folds``: with
one fold the partition is the saved training/validation split.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from wqu.config import CRED_ARCHITECTURE, EVENT_SPLIT_GROUPING, ModelConfig, ProjectConfig, TrainingConfig
from wqu.data.loading import CV_COMPONENTS
from wqu.data.splitting import (
    FOLD_COLUMN,
    SPLIT_COLUMN,
    TRAIN,
    VALIDATION,
    assign_folds,
    attach_column,
    attach_split,
    load_split,
)
from wqu.models.cred import build_model
from wqu.models.loss import bad_class_weight, prior_corrected_probability, weighted_bce_loss
from wqu.models.uncertainty import deterministic_probabilities, prediction_table
from wqu.preprocessing.features import check_representation_features, representation_feature_path
from wqu.preprocessing.pipeline import FEATURE_INDEX_COLUMN, read_feature_records
from wqu.preprocessing.time_masks import drop_records_without_data, read_time_masks, time_mask_path
from wqu.training.data import FeatureDataset, FrequencyStatistics, frequency_statistics, make_loader
from wqu.training.trainer import FitResult, fit, set_seed


@dataclass(frozen=True)
class ExperimentRecords:
    """Records available to an experiment.

    Attributes:
        features_path: Feature file.
        records: Record table with the split column, without the records excluded by time masking.
        excluded: Records without any recorded data (only with time masking).
        time_masks: Time masks of the feature file, or None without time masking.
    """

    features_path: Path
    records: pd.DataFrame
    excluded: pd.DataFrame
    time_masks: np.ndarray | None


@dataclass(frozen=True)
class Partition:
    """Training and validation records of one model, with their training statistics.

    Attributes:
        train_records: Training records.
        validation_records: Validation records.
        statistics: Per-frequency normalization statistics of the training records.
        bad_weight: Weight of the bad class (good / bad records in training).
    """

    train_records: pd.DataFrame
    validation_records: pd.DataFrame
    statistics: FrequencyStatistics
    bad_weight: float


@dataclass(frozen=True)
class Fold:
    """One partition of a cross-validation.

    Attributes:
        number: Fold number (1 to the number of folds); its records are the validation records.
        partition: Training and validation records of the fold with their statistics.
    """

    number: int
    partition: Partition


MASKED_SUFFIX = "_masked"
STATISTICS_FILE = "normalization.npz"
FOLD_DIR_FORMAT = "fold_{}"
RECORD_COLUMN = "record"


def experiment_name(config: ProjectConfig) -> str:
    """Name of the data and network variant of an experiment.

    Args:
        config: Project configuration.

    Returns:
        ``<representation>_<window placement>`` with ``_masked`` for the time-masked network and
        ``_<architecture>`` for architectures other than CRED (CRED names are unchanged).
    """
    suffix = MASKED_SUFFIX if config.model.time_masking else ""
    if config.model.architecture != CRED_ARCHITECTURE:
        suffix += f"_{config.model.architecture}"
    return f"{config.training.representation}_{config.preprocessing.window_placement}{suffix}"


def load_experiment_records(config: ProjectConfig) -> ExperimentRecords:
    """Records of the configured feature file with their split, without the records lacking data.

    Args:
        config: Project configuration (feature source, window placement, time masking, split file).

    Returns:
        ExperimentRecords.

    Raises:
        ValueError: If the split is not grouped by event (the experiments compare models on unseen events).
    """
    if config.split.grouping != EVENT_SPLIT_GROUPING:
        raise ValueError(f"split.grouping is '{config.split.grouping}': the learning curve and the search use the "
                         f"split grouped by event, set it to '{EVENT_SPLIT_GROUPING}'.")
    config_columns = config.columns
    training = config.training
    features_path = representation_feature_path(config.preprocessing, training.feature_source, training.representation)
    check_representation_features(features_path, config.preprocessing, training.representation)
    records = read_feature_records(features_path, config_columns.text)
    records[SPLIT_COLUMN] = attach_split(records, load_split(config.split.output_file, config_columns), config_columns)
    time_masks = read_time_masks(time_mask_path(features_path)) if config.model.time_masking else None
    excluded = records.iloc[:0]
    if time_masks is not None:
        records, excluded = drop_records_without_data(records, time_masks)
    return ExperimentRecords(features_path, records, excluded, time_masks)


def build_partition(
    experiment: ExperimentRecords,
    representation: str,
    train_records: pd.DataFrame,
    validation_records: pd.DataFrame,
    label_column: str,
    statistics_path: Path,
) -> Partition:
    """Partition with normalization statistics and bad-class weight of its training records.

    The statistics are read from ``statistics_path`` if it exists, otherwise computed and saved.

    Args:
        experiment: Output of ``load_experiment_records``.
        representation: ``spectrogram`` or ``scalogram``.
        train_records: Training records.
        validation_records: Validation records.
        label_column: Label column.
        statistics_path: Cache file of the statistics (``.npz``).

    Returns:
        Partition.
    """
    if statistics_path.is_file():
        statistics = FrequencyStatistics.load(statistics_path)
    else:
        statistics = frequency_statistics(
            experiment.features_path, representation, train_records[FEATURE_INDEX_COLUMN].to_numpy()
        )
        statistics_path.parent.mkdir(parents=True, exist_ok=True)
        statistics.save(statistics_path)
    return Partition(train_records, validation_records, statistics, bad_class_weight(train_records[label_column].to_numpy()))


def partition_dataset(
    experiment: ExperimentRecords, representation: str, partition: Partition, subset: pd.DataFrame, label_column: str
) -> FeatureDataset:
    """Dataset of standardized images of some records, with the statistics of a partition.

    Args:
        experiment: Output of ``load_experiment_records``.
        representation: ``spectrogram`` or ``scalogram``.
        partition: Partition whose statistics are used.
        subset: Records of the dataset.
        label_column: Label column.

    Returns:
        FeatureDataset (with the time-mask channel when the experiment uses time masking).
    """
    return FeatureDataset(
        experiment.features_path,
        representation,
        subset[FEATURE_INDEX_COLUMN].to_numpy(),
        subset[label_column].to_numpy(),
        partition.statistics,
        experiment.time_masks,
    )


def train_partition(
    experiment: ExperimentRecords,
    partition: Partition,
    model_config: ModelConfig,
    training: TrainingConfig,
    label_column: str,
    run_dir: Path,
    device: str,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> FitResult:
    """Train one model on a partition (resumable from ``run_dir``) and return the fit result.

    The seed ``training.random_seed`` fixes the initial weights and the order of the batches.

    Args:
        experiment: Output of ``load_experiment_records``.
        partition: Training and validation records with their statistics.
        model_config: Model configuration.
        training: Training configuration (representation, seed, optimizer, early stopping).
        label_column: Label column.
        run_dir: Output folder of the model (checkpoint, history, best weights).
        device: Device name.
        epoch_callback: Optional ``(epoch, validation_loss) -> None`` called after every epoch.

    Returns:
        FitResult of ``wqu.training.trainer.fit``.
    """
    representation = training.representation
    train_loader = make_loader(
        partition_dataset(experiment, representation, partition, partition.train_records, label_column),
        training.batch_size, True, training.num_workers, training.random_seed,
    )
    validation_loader = make_loader(
        partition_dataset(experiment, representation, partition, partition.validation_records, label_column),
        training.batch_size, False, training.num_workers, training.random_seed,
    )
    set_seed(training.random_seed)
    network = build_model(model_config, len(CV_COMPONENTS), len(partition.statistics.mean))
    return fit(
        network, train_loader, validation_loader, weighted_bce_loss(partition.bad_weight), training,
        partition.bad_weight, run_dir, device, epoch_callback=epoch_callback,
    )


def best_epoch_metrics(result: FitResult) -> dict[str, float]:
    """Validation metrics at the best epoch of a fit.

    Args:
        result: Output of ``fit``.

    Returns:
        Dictionary with best epoch, epochs run, validation loss, AUC and ECE at the best epoch,
        and training minutes.
    """
    best = result.history.loc[result.history["epoch"] == result.best_epoch].iloc[0]
    return {
        "best_epoch": int(result.best_epoch),
        "epochs": int(len(result.history)),
        "validation_loss": float(best["validation_loss"]),
        "validation_auc": float(best["validation_auc"]),
        "validation_ece": float(best["validation_ece"]),
        "minutes": float(result.history["seconds"].sum() / 60.0),
    }


def fold_dir(base_dir: Path, fold_number: int, n_folds: int) -> Path:
    """Folder of the outputs of one fold.

    Args:
        base_dir: Folder of the experiment or trial.
        fold_number: Fold number.
        n_folds: Number of folds (1 = the saved split).

    Returns:
        ``base_dir`` with one fold, ``base_dir/fold_<number>`` otherwise.
    """
    return base_dir if n_folds == 1 else base_dir / FOLD_DIR_FORMAT.format(fold_number)


def records_with_folds(config: ProjectConfig, records: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Records with their cross-validation fold.

    The folds are assigned on all the records of the split file (``assign_folds`` with
    ``split.random_seed``), so they do not depend on the records of the feature file or on
    the records excluded by time masking. With one fold, fold 1 is the validation set of the
    split and fold 0 its training set. With as many folds as the split, fold 1 must equal its
    validation set.

    Args:
        config: Project configuration.
        records: Records with the split column (``ExperimentRecords.records``).
        n_folds: Number of folds.

    Returns:
        Copy of ``records`` with the ``fold`` column.

    Raises:
        ValueError: If ``n_folds`` is not positive, or fold 1 differs from the validation set when it should equal it.
    """
    if n_folds < 1:
        raise ValueError(f"The number of folds must be positive, got {n_folds}.")
    with_folds = records.copy()
    if n_folds == 1:
        with_folds[FOLD_COLUMN] = np.where(records[SPLIT_COLUMN] == VALIDATION, 1, 0)
        return with_folds
    folds = assign_folds(load_split(config.split.output_file, config.columns), config.columns, n_folds,
                         config.split.random_seed)
    with_folds[FOLD_COLUMN] = attach_column(records, folds, FOLD_COLUMN, config.columns).astype(int)
    if n_folds == config.split.n_folds:
        mismatched = int(((with_folds[FOLD_COLUMN] == 1) != (with_folds[SPLIT_COLUMN] == VALIDATION)).sum())
        if mismatched:
            raise ValueError(f"Fold 1 differs from the validation set of the split in {mismatched} records.")
    return with_folds


def fold_partitions(
    experiment: ExperimentRecords,
    records: pd.DataFrame,
    n_folds: int,
    representation: str,
    label_column: str,
    statistics_dir: Path,
) -> list[Fold]:
    """Training/validation partitions of a cross-validation, with cached statistics.

    Args:
        experiment: Output of ``load_experiment_records``.
        records: Output of ``records_with_folds``.
        n_folds: Number of folds (1 = the saved split, with its training records only).
        representation: ``spectrogram`` or ``scalogram``.
        label_column: Label column.
        statistics_dir: Folder of the statistics cache (one file per fold, see ``fold_dir``).

    Returns:
        One Fold per validation fold, in order.
    """
    folds = []
    for number in range(1, n_folds + 1):
        validation = records[FOLD_COLUMN] == number
        train = records[SPLIT_COLUMN] == TRAIN if n_folds == 1 else ~validation
        partition = build_partition(
            experiment, representation, records[train], records[validation], label_column,
            fold_dir(statistics_dir, number, n_folds) / STATISTICS_FILE,
        )
        folds.append(Fold(number, partition))
    return folds


def partition_predictions(
    experiment: ExperimentRecords,
    partition: Partition,
    model_config: ModelConfig,
    training: TrainingConfig,
    label_column: str,
    weights_path: Path,
    device: str,
    mc_samples: int | None,
) -> pd.DataFrame:
    """Predictions of a trained model on the validation records of its partition.

    Args:
        experiment: Output of ``load_experiment_records``.
        partition: Partition the model was trained on.
        model_config: Model configuration of the model.
        training: Training configuration (representation, batch size, workers, seed).
        label_column: Label column.
        weights_path: File of the model weights (``best_model.pt``).
        device: Device name.
        mc_samples: MC dropout passes, or None for the deterministic pass only.

    Returns:
        Table with ``record`` (index of the record table), label and the probabilities: deterministic
        and prior-corrected with ``mc_samples`` None, otherwise the columns of ``prediction_table``.

    Raises:
        FileNotFoundError: If the weights file does not exist.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"No model weights at {weights_path}.")
    validation_records = partition.validation_records
    loader = make_loader(
        partition_dataset(experiment, training.representation, partition, validation_records, label_column),
        training.batch_size, False, training.num_workers, training.random_seed,
    )
    network = build_model(model_config, len(CV_COMPONENTS), len(partition.statistics.mean))
    network.load_state_dict(torch.load(weights_path, map_location="cpu"))
    if mc_samples is None:
        probabilities = deterministic_probabilities(network, loader, device)
        table = pd.DataFrame({
            "probability_deterministic": probabilities,
            "probability_deterministic_corrected": prior_corrected_probability(probabilities, partition.bad_weight),
        })
    else:
        table = prediction_table(network, loader, mc_samples, device, partition.bad_weight)
    table.insert(0, RECORD_COLUMN, validation_records.index.to_numpy())
    table.insert(1, label_column, validation_records[label_column].to_numpy())
    return table
