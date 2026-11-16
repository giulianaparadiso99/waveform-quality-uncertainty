"""Hyperparameter search with Optuna, resumable across sessions.

The study is stored in an SQLite database in ``search_dir/<study_name>/``, with one
folder per trial (checkpoint, best model, history). The TPE sampler proposes the
hyperparameters of each trial from the results of the earlier ones; the median
pruner stops a trial whose validation loss is worse than the median of the earlier
trials at the same epoch. Each trial is scored by its best validation loss.

Interrupted sessions: every running trial sends a heartbeat to the database. When a
new session starts, a trial without heartbeat for longer than the grace period is
marked as failed and a copy with the same hyperparameters is queued; the copy resumes
training from the checkpoint of the original trial (the epochs already done are
reported again to the pruner).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import optuna
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback
from optuna.trial import TrialState
from torch import nn
from torch.utils.data import DataLoader

from wqu.config import ModelConfig, SearchConfig, TrainingConfig
from wqu.training.trainer import fit, set_seed

DATABASE_FILE = "study.db"
TRIAL_DIR_FORMAT = "trial_{:03d}"
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
    history = RetryHeartbeatStaleTrialCallback.retry_history(frozen)
    return study_dir(search) / TRIAL_DIR_FORMAT.format(history[0] if history else trial.number)


def pruning_callback(trial: optuna.Trial) -> Callable[[int, float], None]:
    """Epoch callback that reports the validation loss and stops the trial when the pruner says so.

    Args:
        trial: Optuna trial.

    Returns:
        Function ``(epoch, validation_loss) -> None`` raising ``optuna.TrialPruned``.
    """

    def report(epoch: int, validation_loss: float) -> None:
        trial.report(validation_loss, step=epoch)
        if trial.should_prune():
            message = f"Trial {trial.number} pruned at epoch {epoch} (validation loss {validation_loss:.4f})."
            print(message)
            raise optuna.TrialPruned(message)

    return report


def make_objective(
    search: SearchConfig,
    model: ModelConfig,
    training: TrainingConfig,
    build_model: Callable[[ModelConfig], nn.Module],
    loss: nn.Module,
    make_train_loader: Callable[[], DataLoader],
    validation_loader: DataLoader,
    bad_weight: float,
    device: str,
) -> Callable[[optuna.Trial], float]:
    """Objective of the study: train one trial and return its best validation loss.

    Every trial starts from the same seed (``training.random_seed``), so trials differ
    only in their hyperparameters.

    Args:
        search: Search configuration.
        model: Base model configuration.
        training: Base training configuration.
        build_model: Function creating a network from a model configuration.
        loss: Weighted loss module.
        make_train_loader: Function creating a fresh, seeded training loader.
        validation_loader: Validation loader.
        bad_weight: Weight of the bad class used in the loss.
        device: Device name.

    Returns:
        Objective function for ``study.optimize``.
    """

    def objective(trial: optuna.Trial) -> float:
        settings = suggest_settings(trial, search, model, training)
        set_seed(training.random_seed)
        network = build_model(settings.model)
        result = fit(
            network,
            make_train_loader(),
            validation_loader,
            loss,
            settings.training,
            bad_weight,
            trial_dir(search, trial),
            device,
            epoch_callback=pruning_callback(trial),
        )
        best = result.history.loc[result.history["epoch"] == result.best_epoch].iloc[0]
        trial.set_user_attr("best_epoch", int(result.best_epoch))
        trial.set_user_attr("epochs", int(len(result.history)))
        trial.set_user_attr("validation_auc", float(best["validation_auc"]))
        trial.set_user_attr("validation_ece", float(best["validation_ece"]))
        trial.set_user_attr("minutes", float(result.history["seconds"].sum() / 60.0))
        return float(best["validation_loss"])

    return objective


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
