"""Building blocks shared by the experiments that train several models (learning curve, search).

An experiment trains models on one or more training/validation partitions of the records.
For each partition the normalization statistics and the bad-class weight come from its
training records only; the statistics are cached in a file, so that later sessions and
other models on the same partition reuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wqu.config import ModelConfig, ProjectConfig, TrainingConfig
from wqu.data.loading import CV_COMPONENTS
from wqu.data.splitting import SPLIT_COLUMN, attach_split, load_split
from wqu.models.cred import build_model
from wqu.models.loss import bad_class_weight, weighted_bce_loss
from wqu.preprocessing.pipeline import (
    FEATURE_INDEX_COLUMN,
    check_feature_placement,
    feature_file_path,
    read_feature_records,
)
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


MASKED_SUFFIX = "_masked"


def experiment_name(config: ProjectConfig) -> str:
    """Name of the data and network variant of an experiment.

    Args:
        config: Project configuration.

    Returns:
        ``<representation>_<window placement>`` with ``_masked`` for the time-masked network.
    """
    suffix = MASKED_SUFFIX if config.model.time_masking else ""
    return f"{config.training.representation}_{config.preprocessing.window_placement}{suffix}"


def load_experiment_records(config: ProjectConfig) -> ExperimentRecords:
    """Records of the configured feature file with their split, without the records lacking data.

    Args:
        config: Project configuration (feature source, window placement, time masking, split file).

    Returns:
        ExperimentRecords.
    """
    config_columns = config.columns
    features_path = feature_file_path(config.preprocessing, config.training.feature_source)
    check_feature_placement(features_path, config.preprocessing)
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
