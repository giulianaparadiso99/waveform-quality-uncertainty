"""Hyperparameter search with Optuna, resumable across sessions.

The study is stored in an SQLite database in ``search_dir/<study_name>/``, with one
folder per trial (checkpoint, best model, history; one subfolder per fold with
cross-validation). The TPE sampler proposes the hyperparameters of each trial from the
results of the earlier ones. Each trial trains one model per fold, in order, and is
scored by the mean over the folds of the best validation loss.

Pruning with cross-validation: the validation loss of epoch ``e`` of fold ``f`` is
reported at step ``(f - 1) * max_epochs + e``, so the median pruner compares a trial
with the earlier trials at the same epoch of the same fold. The first
``pruner_warmup_epochs`` epochs of every fold are never pruned.

Interrupted sessions: every running trial sends a heartbeat to the database. When a
new session starts, a trial without heartbeat for longer than the grace period is
marked as failed and a copy with the same hyperparameters is queued; the copy resumes
training from the checkpoint of the original trial (the epochs already done are
reported again to the pruner; folds already finished are not trained again).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback
from optuna.trial import TrialState

from wqu.config import ModelConfig, SearchConfig, TrainingConfig
from wqu.data.splitting import FOLD_COLUMN
from wqu.training.experiment import (
    ExperimentRecords,
    Fold,
    best_epoch_metrics,
    fold_dir,
    partition_predictions,
    train_partition,
)
from wqu.training.trainer import BEST_MODEL_FILE

DATABASE_FILE = "study.db"
TRIAL_DIR_FORMAT = "trial_{:03d}"
FOLD_PREDICTIONS_FILE = "fold_predictions.csv"
MC_FOLD_PREDICTIONS_FILE = "fold_predictions_mc.csv"
CROSS_VALIDATION_SUFFIX = "_cv{}"
VALIDATION_LOSS_STD_ATTRIBUTE = "validation_loss_std"
FINISHED_STATES: tuple[TrialState, ...] = (TrialState.COMPLETE, TrialState.PRUNED)


@dataclass(frozen=True)
class TrialSettings:
    """Model and training configuration of one trial.

    Attributes:
        model: Model configuration with the sampled hyperparameters.
        training: Training configuration with the sampled learning rate.
    """

    model: ModelConfig
    training: TrainingConfig


def variant_search(search: SearchConfig, variant: str) -> SearchConfig:
    """Search configuration whose study is specific to a data and network variant and to the folds.

    Searches on different representations, window placements, networks or numbers of folds
    are stored in separate databases and folders.

    Args:
        search: Search configuration (``study_name`` is the prefix).
        variant: Name of the variant (``wqu.training.experiment.experiment_name``).

    Returns:
        Copy of ``search`` with study name ``<study_name>_<variant>``, followed by ``_cv<k>``
        when ``cross_validation_folds`` is above 1.
    """
    folds = search.cross_validation_folds
    suffix = CROSS_VALIDATION_SUFFIX.format(folds) if folds > 1 else ""
    return replace(search, study_name=f"{search.study_name}_{variant}{suffix}")


def study_dir(search: SearchConfig) -> Path:
    """Folder of the study (database and trial folders).

    Args:
        search: Search configuration.

    Returns:
        ``search_dir/<study_name>``.
    """
    return search.search_dir / search.study_name


def load_study(search: SearchConfig) -> optuna.Study:
    """Create the study, or load it if it already exists.

    The seed of the sampler is ``sampler_seed`` plus the number of trials already in the
    study: a resumed session does not repeat the random proposals of the first one, and
    the sequence of proposals stays reproducible.

    Args:
        search: Search configuration.

    Returns:
        Optuna study minimizing the validation loss.
    """
    folder = study_dir(search)
    folder.mkdir(parents=True, exist_ok=True)
    storage = RDBStorage(
        url=f"sqlite:///{folder / DATABASE_FILE}",
        heartbeat_interval=search.heartbeat_interval_s,
        grace_period=search.grace_period_s,
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=search.max_retries),
    )
    study = optuna.create_study(
        study_name=search.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=search.pruner_startup_trials, n_warmup_steps=search.pruner_warmup_epochs
        ),
    )
    study.sampler = optuna.samplers.TPESampler(seed=search.sampler_seed + len(study.get_trials(deepcopy=False)))
    return study


BASELINE_ATTRIBUTE = "baseline"


def baseline_params(search: SearchConfig, model: ModelConfig, training: TrainingConfig) -> dict:
    """Search parameters reproducing the configured model and training settings.

    Args:
        search: Search configuration (search space).
        model: Model configuration.
        training: Training configuration.

    Returns:
        Parameters in the form sampled by ``suggest_settings``.

    Raises:
        ValueError: If the configuration lies outside the search space.
    """
    problems = []
    if model.conv_filters not in search.conv_filters_choices:
        problems.append(f"conv_filters {model.conv_filters} not in {search.conv_filters_choices}")
    if model.conv_dropout not in search.conv_dropout_choices:
        problems.append(f"conv dropout {model.conv_dropout} not in {search.conv_dropout_choices}")
    if model.residual_dropout not in (0.0, model.conv_dropout):
        problems.append("residual dropout must be 0 or equal to the conv dropout")
    if (model.residual_dropout > 0) not in search.residual_dropout_options:
        problems.append("residual dropout setting not in the search space")
    if model.recurrent_units not in search.recurrent_units_choices:
        problems.append(f"recurrent_units {model.recurrent_units} not in {search.recurrent_units_choices}")
    bounds = {
        "recurrent dropout": (model.recurrent_dropout, search.recurrent_dropout_range),
        "head dropout": (model.head_dropout, search.head_dropout_range),
        "learning rate": (training.learning_rate, search.learning_rate_range),
    }
    for name, (value, (low, high)) in bounds.items():
        if not low <= value <= high:
            problems.append(f"{name} {value} outside [{low}, {high}]")
    if problems:
        raise ValueError("The baseline configuration is outside the search space: " + "; ".join(problems) + ".")
    return {
        "conv_dropout": model.conv_dropout,
        "residual_dropout": model.residual_dropout > 0,
        "conv_filters": search.conv_filters_choices.index(model.conv_filters),
        "recurrent_units": model.recurrent_units,
        "recurrent_dropout": model.recurrent_dropout,
        "head_dropout": model.head_dropout,
        "learning_rate": training.learning_rate,
    }


def enqueue_baseline_trial(study: optuna.Study, search: SearchConfig, model: ModelConfig, training: TrainingConfig) -> bool:
    """Queue the configured settings as the first trial of a new study.

    Args:
        study: Optuna study.
        search: Search configuration.
        model: Model configuration of the baseline.
        training: Training configuration of the baseline.

    Returns:
        True if the trial was queued (only when ``search.enqueue_baseline`` is set and the
        study has no trials yet).
    """
    if not search.enqueue_baseline or study.get_trials(deepcopy=False):
        return False
    study.enqueue_trial(baseline_params(search, model, training), user_attrs={BASELINE_ATTRIBUTE: True})
    return True


def finished_trials(study: optuna.Study) -> int:
    """Number of completed or pruned trials.

    Args:
        study: Optuna study.

    Returns:
        Count of finished trials.
    """
    return len(study.get_trials(deepcopy=False, states=FINISHED_STATES))


def suggest_settings(
    trial: optuna.Trial, search: SearchConfig, model: ModelConfig, training: TrainingConfig
) -> TrialSettings:
    """Sample the hyperparameters of a trial and apply them to the base configurations.

    Args:
        trial: Optuna trial.
        search: Search configuration (search space).
        model: Base model configuration.
        training: Base training configuration.

    Returns:
        TrialSettings.
    """
    conv_dropout = trial.suggest_categorical("conv_dropout", list(search.conv_dropout_choices))
    residual = trial.suggest_categorical("residual_dropout", list(search.residual_dropout_options))
    filters_index = trial.suggest_categorical("conv_filters", list(range(len(search.conv_filters_choices))))
    sampled_model = replace(
        model,
        conv_filters=search.conv_filters_choices[filters_index],
        recurrent_units=trial.suggest_categorical("recurrent_units", list(search.recurrent_units_choices)),
        conv_dropout=conv_dropout,
        recurrent_dropout=trial.suggest_float("recurrent_dropout", *search.recurrent_dropout_range),
        head_dropout=trial.suggest_float("head_dropout", *search.head_dropout_range),
        residual_dropout=conv_dropout if residual else 0.0,
    )
    sampled_training = replace(
        training, learning_rate=trial.suggest_float("learning_rate", *search.learning_rate_range, log=True)
    )
    return TrialSettings(model=sampled_model, training=sampled_training)


def trial_dir(search: SearchConfig, trial: optuna.Trial) -> Path:
    """Folder of a trial; a retried trial reuses the folder of the trial it continues.

    Args:
        search: Search configuration.
        trial: Optuna trial.

    Returns:
        ``<study folder>/trial_<number>`` of the first trial in the retry chain.
    """
    frozen = next(stored for stored in trial.study.get_trials(deepcopy=False) if stored.number == trial.number)
    return frozen_trial_dir(search, frozen)


def frozen_trial_dir(search: SearchConfig, frozen: optuna.trial.FrozenTrial) -> Path:
    """Folder of a stored trial; a retried trial uses the folder of the trial it continues.

    Args:
        search: Search configuration.
        frozen: Stored trial (e.g. ``study.best_trial``).

    Returns:
        ``<study folder>/trial_<number>`` of the first trial in the retry chain.
    """
    history = RetryHeartbeatStaleTrialCallback.retry_history(frozen)
    return study_dir(search) / TRIAL_DIR_FORMAT.format(history[0] if history else frozen.number)


def trial_settings(params: dict, search: SearchConfig, model: ModelConfig, training: TrainingConfig) -> TrialSettings:
    """Model and training configuration of stored trial parameters.

    Args:
        params: Parameters of a trial (``FrozenTrial.params``).
        search: Search configuration.
        model: Base model configuration.
        training: Base training configuration.

    Returns:
        TrialSettings, as built by ``suggest_settings`` for that trial.
    """
    return suggest_settings(optuna.trial.FixedTrial(params), search, model, training)


def pruning_step(fold_number: int, epoch: int, max_epochs: int) -> int:
    """Step at which the validation loss of an epoch of a fold is reported to the pruner.

    Args:
        fold_number: Fold number (from 1).
        epoch: Epoch (from 1).
        max_epochs: Largest number of epochs of a fold.

    Returns:
        ``(fold_number - 1) * max_epochs + epoch``.
    """
    return (fold_number - 1) * max_epochs + epoch


def pruning_callback(
    trial: optuna.Trial, fold_number: int, max_epochs: int, warmup_epochs: int
) -> Callable[[int, float], None]:
    """Epoch callback that reports the validation loss and stops the trial when the pruner says so.

    Args:
        trial: Optuna trial.
        fold_number: Fold being trained.
        max_epochs: Largest number of epochs of a fold.
        warmup_epochs: Epochs of every fold that are never pruned.

    Returns:
        Function ``(epoch, validation_loss) -> None`` raising ``optuna.TrialPruned``.
    """

    def report(epoch: int, validation_loss: float) -> None:
        trial.report(validation_loss, step=pruning_step(fold_number, epoch, max_epochs))
        if epoch > warmup_epochs and trial.should_prune():
            message = (f"Trial {trial.number} pruned at epoch {epoch} of fold {fold_number} "
                       f"(validation loss {validation_loss:.4f}).")
            print(message)
            raise optuna.TrialPruned(message)

    return report


def record_fold_metrics(trial: optuna.Trial, fold_number: int, n_folds: int, metrics: dict[str, float]) -> None:
    """Store the best-epoch metrics of a finished fold as user attributes of the trial.

    Args:
        trial: Optuna trial.
        fold_number: Fold number.
        n_folds: Number of folds (with one fold the attributes have no fold prefix).
        metrics: Output of ``best_epoch_metrics``.
    """
    prefix = f"fold_{fold_number}_" if n_folds > 1 else ""
    for name, value in metrics.items():
        trial.set_user_attr(f"{prefix}{name}", value)


def summary_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Summary over the folds of the best-epoch metrics of a trial.

    Args:
        fold_metrics: Output of ``best_epoch_metrics`` for every fold.

    Returns:
        Mean validation loss, its standard deviation over the folds, mean AUC, ECE and best
        epoch, and total epochs and minutes.
    """
    losses = np.array([metrics["validation_loss"] for metrics in fold_metrics])
    return {
        "validation_loss": float(losses.mean()),
        VALIDATION_LOSS_STD_ATTRIBUTE: float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
        "validation_auc": float(np.mean([metrics["validation_auc"] for metrics in fold_metrics])),
        "validation_ece": float(np.mean([metrics["validation_ece"] for metrics in fold_metrics])),
        "best_epoch": float(np.mean([metrics["best_epoch"] for metrics in fold_metrics])),
        "epochs": float(sum(metrics["epochs"] for metrics in fold_metrics)),
        "minutes": float(sum(metrics["minutes"] for metrics in fold_metrics)),
    }


def save_fold_predictions(
    experiment: ExperimentRecords,
    fold: Fold,
    settings: TrialSettings,
    label_column: str,
    run_dir: Path,
    device: str,
) -> Path:
    """Write the deterministic predictions of a finished fold, unless already written.

    Args:
        experiment: Records of the experiment.
        fold: Fold the model was trained on.
        settings: Settings of the trial.
        label_column: Label column.
        run_dir: Folder of the fold (with ``best_model.pt``).
        device: Device name.

    Returns:
        Path of the predictions file.
    """
    path = run_dir / FOLD_PREDICTIONS_FILE
    if not path.is_file():
        predictions = partition_predictions(
            experiment, fold.partition, settings.model, settings.training, label_column,
            run_dir / BEST_MODEL_FILE, device, mc_samples=None,
        )
        predictions.to_csv(path, index=False)
    return path


def make_objective(
    search: SearchConfig,
    model: ModelConfig,
    training: TrainingConfig,
    experiment: ExperimentRecords,
    folds: list[Fold],
    label_column: str,
    device: str,
) -> Callable[[optuna.Trial], float]:
    """Objective of the study: train one trial on every fold and return the mean best validation loss.

    Every model starts from the same seed (``training.random_seed``), so trials differ only in
    their hyperparameters. The metrics of every finished fold are stored as user attributes
    (``fold_<k>_...`` with more than one fold), with their summary over the folds at the end.

    Args:
        search: Search configuration.
        model: Base model configuration.
        training: Base training configuration.
        experiment: Records of the experiment (``load_experiment_records``).
        folds: Partitions of the cross-validation (``fold_partitions``).
        label_column: Label column.
        device: Device name.

    Returns:
        Objective function for ``study.optimize``.
    """

    def objective(trial: optuna.Trial) -> float:
        settings = suggest_settings(trial, search, model, training)
        folder = trial_dir(search, trial)
        fold_metrics = []
        for fold in folds:
            run_dir = fold_dir(folder, fold.number, len(folds))
            if len(folds) > 1:
                print(f"Trial {trial.number}, fold {fold.number} of {len(folds)}")
            result = train_partition(
                experiment, fold.partition, settings.model, settings.training, label_column, run_dir, device,
                epoch_callback=pruning_callback(trial, fold.number, training.max_epochs, search.pruner_warmup_epochs),
            )
            metrics = best_epoch_metrics(result)
            record_fold_metrics(trial, fold.number, len(folds), metrics)
            fold_metrics.append(metrics)
            if search.save_fold_predictions:
                save_fold_predictions(experiment, fold, settings, label_column, run_dir, device)
        summary = summary_metrics(fold_metrics)
        for name, value in summary.items():
            if name != "validation_loss":
                trial.set_user_attr(name, value)
        return summary["validation_loss"]

    return objective


def out_of_fold_predictions(
    search: SearchConfig,
    frozen: optuna.trial.FrozenTrial,
    model: ModelConfig,
    training: TrainingConfig,
    experiment: ExperimentRecords,
    folds: list[Fold],
    label_column: str,
    device: str,
    mc_samples: int,
) -> pd.DataFrame:
    """MC dropout predictions of a completed trial, each record predicted by the model of its fold.

    The table of every fold is cached in its folder (``fold_predictions_mc.csv``).

    Args:
        search: Search configuration.
        frozen: Completed trial.
        model: Base model configuration.
        training: Base training configuration.
        experiment: Records of the experiment.
        folds: Partitions of the cross-validation used by the search.
        label_column: Label column.
        device: Device name.
        mc_samples: MC dropout passes.

    Returns:
        Table of ``partition_predictions`` for all the validation records of the folds, with
        the ``fold`` column.

    Raises:
        ValueError: If the trial is not complete.
    """
    if frozen.state != TrialState.COMPLETE:
        raise ValueError(f"Trial {frozen.number} is not complete ({frozen.state.name}).")
    settings = trial_settings(frozen.params, search, model, training)
    tables = []
    for fold in folds:
        run_dir = fold_dir(frozen_trial_dir(search, frozen), fold.number, len(folds))
        path = run_dir / MC_FOLD_PREDICTIONS_FILE
        if path.is_file():
            table = pd.read_csv(path)
        else:
            table = partition_predictions(
                experiment, fold.partition, settings.model, settings.training, label_column,
                run_dir / BEST_MODEL_FILE, device, mc_samples,
            )
            table.to_csv(path, index=False)
        table.insert(2, FOLD_COLUMN, fold.number)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def run_search(study: optuna.Study, objective: Callable[[optuna.Trial], float], n_trials: int) -> int:
    """Run trials until ``n_trials`` are finished (completed or pruned), counting earlier sessions.

    Args:
        study: Optuna study.
        objective: Output of ``make_objective``.
        n_trials: Trials to finish in total.

    Returns:
        Number of trials run in this session.
    """
    started = 0
    while finished_trials(study) < n_trials:
        study.optimize(objective, n_trials=1, gc_after_trial=True)
        started += 1
    return started
