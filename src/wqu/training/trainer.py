"""Training loop with learning-rate reduction, early stopping and checkpointing.

At every epoch the history records the training loss and metrics (from the
outputs computed while training, with dropout active), the validation loss and
metrics (evaluation mode, dropout off), the learning rate and the duration. The
model with the lowest validation loss is saved and restored at the end.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from wqu.config import TrainingConfig
from wqu.models.loss import prior_corrected_probability
from wqu.training.metrics import classification_metrics, expected_calibration_error

BEST_MODEL_FILE = "best_model.pt"
HISTORY_FILE = "history.csv"
CONFIG_COPY_FILE = "config.yaml"
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


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss: nn.Module,
    training: TrainingConfig,
    bad_weight: float,
    run_dir: Path,
    device: str,
) -> FitResult:
    """Train with Adam, learning-rate reduction on plateau and early stopping on the validation loss.

    The history is written to ``run_dir/history.csv`` after every epoch, so it can be
    inspected while training; the best weights are written to ``run_dir/best_model.pt``
    and loaded into ``model`` at the end.

    Args:
        model: Network.
        train_loader: Training loader (shuffled).
        validation_loader: Validation loader.
        loss: Weighted loss module.
        training: Training configuration.
        bad_weight: Weight of the bad class used in the loss.
        run_dir: Output folder of the run.
        device: Device name.

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
    best_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    rows = []
    for epoch in range(1, training.max_epochs + 1):
        start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss, train_logits, train_labels = train_one_epoch(model, train_loader, loss, optimizer, device)
        validation_loss, validation_logits, validation_labels = evaluate(model, validation_loader, loss, device)
        scheduler.step(validation_loss)
        train_metrics = epoch_metrics(train_logits, train_labels, bad_weight, training.calibration_bins)
        validation_metrics = epoch_metrics(validation_logits, validation_labels, bad_weight, training.calibration_bins)
        rows.append(
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
        pd.DataFrame(rows).to_csv(run_dir / HISTORY_FILE, index=False)
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, epochs_without_improvement = validation_loss, epoch, 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:3d}  train loss {train_loss:.4f}  validation loss {validation_loss:.4f}  "
            f"validation AUC {validation_metrics['auc']:.4f}  lr {learning_rate:.1e}  "
            f"{rows[-1]['seconds']:.0f} s{'  (best)' if improved else ''}"
        )
        if epochs_without_improvement >= training.early_stopping_patience:
            break
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return FitResult(
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_model_path=best_model_path,
        stopped_early=epochs_without_improvement >= training.early_stopping_patience,
    )
