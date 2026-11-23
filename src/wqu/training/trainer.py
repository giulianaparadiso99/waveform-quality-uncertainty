"""Training loop with learning-rate reduction, early stopping and checkpointing.

At every epoch the history records the training loss and metrics (from the
outputs computed while training, with dropout active), the validation loss and
metrics (evaluation mode, dropout off), the learning rate and the duration. The
model with the lowest validation loss is saved and restored at the end; a new
minimum counts only if it improves on the previous one by ``early_stopping_min_delta``.

Resuming: at the end of every epoch a checkpoint stores the model, the optimizer,
the learning-rate scheduler, the early-stopping state, the history and the state of
the random generators (Python, NumPy, PyTorch, the device and the batch order), so a
run that is interrupted and resumed gives the same result as an uninterrupted one.

Probabilities: the network is trained with the bad class weighted by ``w``, so its
raw probabilities refer to balanced classes. The balanced accuracy uses them with
threshold 0.5 (equal cost of the two errors); the calibration error uses the
probabilities corrected to the class frequencies of the training data
(``prior_corrected_probability``). The AUC does not depend on the correction.
"""

from __future__ import annotations

import random
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from wqu.config import DEFAULT_TIME_MASKING, DEFAULT_WINDOW_PLACEMENT, EVENT_SPLIT_GROUPING, TrainingConfig
from wqu.models.loss import prior_corrected_probability
from wqu.training.metrics import classification_metrics, expected_calibration_error

BEST_MODEL_FILE = "best_model.pt"
HISTORY_FILE = "history.csv"
CONFIG_COPY_FILE = "config.yaml"
CHECKPOINT_FILE = "checkpoint.pt"
RESUME_SECTIONS: tuple[str, ...] = ("model", "training")
RESUME_IGNORED_KEYS: tuple[str, ...] = ("resume",)
WINDOW_PLACEMENT_SETTING = "window_placement"
SPLIT_GROUPING_SETTING = "split_grouping"
VALIDATION_PREDICTIONS_FILE = "validation_predictions.csv"
LATEST_RUN = "latest"
RUN_TIME_FORMAT = "%Y%m%d_%H%M%S"


@dataclass(frozen=True)
class FitResult:
    """Outcome of a training run.

    Attributes:
        history: One row per epoch.
        best_epoch: Epoch with the lowest validation loss (1-based).
        best_model_path: File with the weights of that epoch.
        stopped_early: Whether early stopping ended the run.
    """

    history: pd.DataFrame
    best_epoch: int
    best_model_path: Path
    stopped_early: bool


def select_device(name: str) -> str:
    """Resolve the configured device.

    Args:
        name: ``auto``, ``cpu``, ``mps`` or ``cuda``.

    Returns:
        Device name; ``auto`` picks the Apple GPU, then CUDA, then the CPU.

    Raises:
        RuntimeError: If the requested accelerator is not available.
    """
    available = {"cpu": True, "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available()}
    if name == "auto":
        return "mps" if available["mps"] else "cuda" if available["cuda"] else "cpu"
    if not available[name]:
        raise RuntimeError(f"Device '{name}' is not available on this machine.")
    return name


def create_run_dir(runs_dir: Path, representation: str, config_path: Path) -> Path:
    """Create the folder of a new run and store a copy of the configuration in it.

    Args:
        runs_dir: Parent folder of all runs.
        representation: Input representation, used in the folder name.
        config_path: Configuration file used for the run.

    Returns:
        Path ``runs_dir/<representation>_<date>_<time>``.

    Raises:
        FileExistsError: If a run with the same name already exists.
    """
    run_dir = runs_dir / f"{representation}_{datetime.now().strftime(RUN_TIME_FORMAT)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / CONFIG_COPY_FILE)
    return run_dir


def latest_unfinished_run(runs_dir: Path, representation: str) -> Path | None:
    """Most recent run of a representation with a checkpoint but no validation predictions.

    Args:
        runs_dir: Parent folder of all runs.
        representation: Input representation.

    Returns:
        Path of the run, or None if every run is complete.
    """
    candidates = sorted(
        path
        for path in runs_dir.glob(f"{representation}_*")
        if (path / CHECKPOINT_FILE).is_file() and not (path / VALIDATION_PREDICTIONS_FILE).is_file()
    )
    return candidates[-1] if candidates else None


def _read_yaml(config_path: Path) -> dict:
    """Parsed content of a YAML configuration file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _window_placement(raw: dict) -> str:
    """Window placement of a parsed configuration (the default for files written before the option)."""
    return str(raw["preprocessing"]["window"].get("placement", DEFAULT_WINDOW_PLACEMENT))


def _split_grouping(raw: dict) -> str:
    """Grouping of the split of a parsed configuration (by event for files written before the option)."""
    return str(raw.get("split", {}).get("grouping", EVENT_SPLIT_GROUPING))


def config_window_placement(config_path: Path) -> str:
    """Window placement of the preprocessing named in a configuration file.

    Args:
        config_path: YAML configuration file (for example the copy stored in a run folder).

    Returns:
        Placement name.
    """
    return _window_placement(_read_yaml(config_path))


def config_time_masking(config_path: Path) -> bool:
    """Whether a configuration file uses the time-masked network.

    Args:
        config_path: YAML configuration file (for example the copy stored in a run folder).

    Returns:
        ``model.time_masking`` (the default for files written before the option).
    """
    return bool(_read_yaml(config_path)["model"].get("time_masking", DEFAULT_TIME_MASKING))


def training_settings(config_path: Path) -> dict:
    """Settings that must not change when a run is resumed.

    Args:
        config_path: YAML configuration file.

    Returns:
        Dictionary with the ``model`` and ``training`` sections (without the keys irrelevant
        to resuming), the window placement of the features and the grouping of the split.
    """
    raw = _read_yaml(config_path)
    settings = {section: dict(raw[section]) for section in RESUME_SECTIONS}
    for key in RESUME_IGNORED_KEYS:
        settings["training"].pop(key, None)
    settings[WINDOW_PLACEMENT_SETTING] = _window_placement(raw)
    settings[SPLIT_GROUPING_SETTING] = _split_grouping(raw)
    return settings


def prepare_run_dir(runs_dir: Path, representation: str, config_path: Path, resume: bool) -> tuple[Path, bool]:
    """Folder of the run: the most recent unfinished one when resuming, otherwise a new one.

    Args:
        runs_dir: Parent folder of all runs.
        representation: Input representation.
        config_path: Configuration file of the run.
        resume: Whether to resume an unfinished run.

    Returns:
        Tuple (run folder, whether it is resumed).

    Raises:
        ValueError: If the unfinished run was started with different model, training, window placement or split
            settings.
    """
    unfinished = latest_unfinished_run(runs_dir, representation) if resume else None
    if unfinished is None:
        return create_run_dir(runs_dir, representation, config_path), False
    if training_settings(unfinished / CONFIG_COPY_FILE) != training_settings(config_path):
        raise ValueError(
            f"{unfinished.name} was started with other model, training, window placement or split settings: "
            "restore them, or set training.resume to false to start a new run."
        )
    return unfinished, True


def find_run_dir(runs_dir: Path, representation: str, run: str) -> Path:
    """Folder of a completed run (one with validation predictions).

    Args:
        runs_dir: Parent folder of all runs.
        representation: Representation of the run, used when ``run`` is ``latest``.
        run: Folder name, or ``latest`` for the most recent run of ``representation``.

    Returns:
        Path of the run folder.

    Raises:
        FileNotFoundError: If the run does not exist or has no validation predictions.
    """
    if run == LATEST_RUN:
        candidates = sorted(
            path for path in runs_dir.glob(f"{representation}_*") if (path / VALIDATION_PREDICTIONS_FILE).is_file()
        )
        if not candidates:
            raise FileNotFoundError(f"No completed {representation} run in {runs_dir}.")
        return candidates[-1]
    run_dir = runs_dir / run
    if not (run_dir / VALIDATION_PREDICTIONS_FILE).is_file():
        raise FileNotFoundError(f"{run_dir} does not contain {VALIDATION_PREDICTIONS_FILE}.")
    return run_dir


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, loss: nn.Module, optimizer: torch.optim.Optimizer, device: str
) -> tuple[float, np.ndarray, np.ndarray]:
    """One pass over the training data.

    Args:
        model: Network.
        loader: Training loader.
        loss: Loss module (on the same device as the model).
        optimizer: Optimizer.
        device: Device name.

    Returns:
        Tuple (mean loss over records, logits, labels) of the epoch.
    """
    model.train()
    total, count, logits, labels = 0.0, 0, [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = loss(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        total += float(batch_loss.detach()) * len(targets)
        count += len(targets)
        logits.append(outputs.detach().cpu())
        labels.append(targets.cpu())
    return total / count, torch.cat(logits).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss: nn.Module, device: str) -> tuple[float, np.ndarray, np.ndarray]:
    """Deterministic pass (evaluation mode) over a data set.

    Args:
        model: Network.
        loader: Loader of the data set.
        loss: Loss module.
        device: Device name.

    Returns:
        Tuple (mean loss over records, logits, labels).
    """
    model.eval()
    total, count, logits, labels = 0.0, 0, [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total += float(loss(outputs, targets)) * len(targets)
        count += len(targets)
        logits.append(outputs.cpu())
        labels.append(targets.cpu())
    return total / count, torch.cat(logits).numpy(), torch.cat(labels).numpy()


def epoch_metrics(logits: np.ndarray, labels: np.ndarray, bad_weight: float, n_bins: int) -> dict[str, float]:
    """AUC and balanced accuracy on the raw probabilities, ECE on the prior-corrected ones.

    Args:
        logits: Network outputs.
        labels: Labels (1 = bad).
        bad_weight: Weight of the bad class used in training.
        n_bins: Bins of the calibration error.

    Returns:
        Dictionary with ``auc``, ``balanced_accuracy`` and ``ece``.
    """
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    metrics = classification_metrics(probabilities, labels, n_bins)
    metrics["ece"] = expected_calibration_error(prior_corrected_probability(probabilities, bad_weight), labels, n_bins)
    return metrics


def random_states(loader: DataLoader, device: str) -> dict:
    """State of every random generator used in training.

    Args:
        loader: Training loader (its generator sets the batch order).
        device: Device name.

    Returns:
        Dictionary of generator states.
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader": loader.generator.get_state() if loader.generator is not None else None,
    }
    if device == "mps":
        states["device"] = torch.mps.get_rng_state()
    elif device == "cuda":
        states["device"] = torch.cuda.get_rng_state()
    return states


def restore_random_states(states: dict, loader: DataLoader, device: str) -> None:
    """Restore the generator states saved by ``random_states``.

    Args:
        states: Output of ``random_states``.
        loader: Training loader.
        device: Device name (must be the one of the saved states).
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["loader"] is not None and loader.generator is not None:
        loader.generator.set_state(states["loader"])
    if device == "mps" and "device" in states:
        torch.mps.set_rng_state(states["device"])
    elif device == "cuda" and "device" in states:
        torch.cuda.set_rng_state(states["device"])


def save_checkpoint(path: Path, state: dict) -> None:
    """Write a checkpoint atomically (temporary file, then rename).

    Args:
        path: Checkpoint file.
        state: Objects to store.
    """
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict:
    """Read a checkpoint written by ``save_checkpoint`` (a trusted file of this project).

    Args:
        path: Checkpoint file.

    Returns:
        Stored objects.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss: nn.Module,
    training: TrainingConfig,
    bad_weight: float,
    run_dir: Path,
    device: str,
    epoch_callback: Callable[[int, float], None] | None = None,
) -> FitResult:
    """Train with Adam, learning-rate reduction on plateau and early stopping on the validation loss; resumable.

    The history is written to ``run_dir/history.csv`` and a checkpoint to
    ``run_dir/checkpoint.pt`` after every epoch; the best weights are written to
    ``run_dir/best_model.pt`` and loaded into ``model`` at the end. If ``run_dir``
    already holds a checkpoint, training continues from the epoch after it (or returns
    at once if that run was finished).

    Args:
        model: Network.
        train_loader: Training loader (shuffled, with its own generator).
        validation_loader: Validation loader.
        loss: Weighted loss module.
        training: Training configuration.
        bad_weight: Weight of the bad class used in the loss.
        run_dir: Output folder of the run.
        device: Device name.
        epoch_callback: Called as ``epoch_callback(epoch, validation_loss)`` after every epoch,
            also for the epochs restored from a checkpoint (e.g. to report to a pruner, which
            may stop the run by raising an exception).

    Returns:
        FitResult.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    loss = loss.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=training.lr_factor, patience=training.lr_patience
    )
    best_model_path = run_dir / BEST_MODEL_FILE
    checkpoint_path = run_dir / CHECKPOINT_FILE
    state = {"best_loss": float("inf"), "best_epoch": 0, "epochs_without_improvement": 0, "rows": [], "finished": False}
    if checkpoint_path.is_file():
        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_random_states(checkpoint["random_states"], train_loader, device)
        state = checkpoint["state"]
        print(f"Resumed after epoch {len(state['rows'])} of {run_dir.name}")
    if epoch_callback is not None:
        for row in state["rows"]:
            epoch_callback(row["epoch"], row["validation_loss"])
    first_epoch = len(state["rows"]) + 1
    for epoch in range(first_epoch, training.max_epochs + 1):
        if state["finished"]:
            break
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_logits, train_labels = train_one_epoch(model, train_loader, loss, optimizer, device)
        validation_loss, validation_logits, validation_labels = evaluate(model, validation_loader, loss, device)
        scheduler.step(validation_loss)
        train_metrics = epoch_metrics(train_logits, train_labels, bad_weight, training.calibration_bins)
        validation_metrics = epoch_metrics(validation_logits, validation_labels, bad_weight, training.calibration_bins)
        state["rows"].append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in validation_metrics.items()},
                "seconds": time.perf_counter() - start,
            }
        )
        pd.DataFrame(state["rows"]).to_csv(run_dir / HISTORY_FILE, index=False)
        improved = validation_loss < state["best_loss"] - training.early_stopping_min_delta
        if improved:
            state.update(best_loss=validation_loss, best_epoch=epoch, epochs_without_improvement=0)
            torch.save(model.state_dict(), best_model_path)
        else:
            state["epochs_without_improvement"] += 1
        state["finished"] = (
            state["epochs_without_improvement"] >= training.early_stopping_patience or epoch == training.max_epochs
        )
        save_checkpoint(
            checkpoint_path,
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "random_states": random_states(train_loader, device),
                "state": state,
            },
        )
        print(
            f"epoch {epoch:3d}  train loss {train_loss:.4f}  validation loss {validation_loss:.4f}  "
            f"validation AUC {validation_metrics['auc']:.4f}  lr {learning_rate:.1e}  "
            f"{state['rows'][-1]['seconds']:.0f} s{'  (best)' if improved else ''}"
        )
        if epoch_callback is not None:
            epoch_callback(epoch, validation_loss)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return FitResult(
        history=pd.DataFrame(state["rows"]),
        best_epoch=state["best_epoch"],
        best_model_path=best_model_path,
        stopped_early=state["epochs_without_improvement"] >= training.early_stopping_patience,
    )
