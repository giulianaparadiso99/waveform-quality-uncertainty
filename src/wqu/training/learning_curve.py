"""Learning curve of the network and choice of the number of cross-validation folds.

With k folds every model is trained on (k - 1) / k of the records, so a small k gives
models trained on fewer records and a pessimistic estimate of the validation loss. The
learning curve measures how the best validation loss depends on the number of training
records: it is fitted with the power law ``L(n) = a + b * n ** (-c)`` (``a`` is the loss
with unlimited data) and evaluated at the training size of each candidate k. The loss
increase of k folds is ``L(n_k) - L(n_all)``, where ``n_all`` is the number of records
available; the suggested k is the smallest one whose increase is within the tolerance,
by default the difference between two runs that differ only in the random seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# Training sizes are divided by this scale before the fit, for numerical stability.
SIZE_SCALE = 1.0e4
MAX_FIT_EVALUATIONS = 20000


@dataclass(frozen=True)
class LearningCurve:
    """Power law ``L(n) = asymptote + amplitude * (n / SIZE_SCALE) ** (-exponent)``.

    Attributes:
        asymptote: Loss with unlimited training records.
        amplitude: Loss in excess of the asymptote at ``SIZE_SCALE`` records.
        exponent: Decay exponent.
    """

    asymptote: float
    amplitude: float
    exponent: float

    def loss(self, n_records: np.ndarray | float) -> np.ndarray:
        """Loss predicted for a number of training records."""
        return self.asymptote + self.amplitude * (np.asarray(n_records, dtype=float) / SIZE_SCALE) ** (-self.exponent)


def training_share_for_folds(n_folds: int, validation_fraction: float) -> float:
    """Share of the current training set equal to the training set of k-fold cross-validation.

    Args:
        n_folds: Number of folds k.
        validation_fraction: Share of the records in the validation set of the current split.

    Returns:
        ``((k - 1) / k) / (1 - validation_fraction)`` (above 1 when k-fold training sets are larger
        than the current one).
    """
    return ((n_folds - 1) / n_folds) / (1.0 - validation_fraction)


def subsample_events(records: pd.DataFrame, share: float, event_column: str, seed: int) -> pd.DataFrame:
    """Records of randomly chosen events, about ``share`` of the records.

    Events are taken in a random order until the chosen records reach ``share`` of the total,
    so that all the records of an event are kept or dropped together. The order depends only
    on ``seed``, so a smaller share is a subset of a larger one.

    Args:
        records: Records with an event column.
        share: Target share of records, in (0, 1].
        event_column: Event identifier column.
        seed: Seed of the random order of the events.

    Returns:
        Subset of ``records``.

    Raises:
        ValueError: If ``share`` is not in (0, 1].
    """
    if not 0.0 < share <= 1.0:
        raise ValueError(f"share must be in (0, 1], got {share}.")
    if share == 1.0:
        return records
    counts = records[event_column].value_counts().sort_index()
    order = np.random.default_rng(seed).permutation(len(counts))
    cumulative = counts.iloc[order].cumsum().to_numpy()
    n_events = int(np.searchsorted(cumulative, share * len(records)) + 1)
    chosen = counts.index[order[:n_events]]
    return records[records[event_column].isin(chosen)]


def fit_learning_curve(n_records: np.ndarray, losses: np.ndarray) -> LearningCurve:
    """Least-squares fit of the power law to (training records, best validation loss) points.

    Args:
        n_records: Training records of each run.
        losses: Best validation loss of each run.

    Returns:
        LearningCurve with a non-negative amplitude and exponent.

    Raises:
        ValueError: If fewer than three distinct training sizes are given.
    """
    sizes = np.asarray(n_records, dtype=float) / SIZE_SCALE
    values = np.asarray(losses, dtype=float)
    if len(np.unique(sizes)) < 3:
        raise ValueError("The power law needs at least three distinct training sizes.")

    def power_law(size: np.ndarray, asymptote: float, amplitude: float, exponent: float) -> np.ndarray:
        return asymptote + amplitude * size ** (-exponent)

    initial = (float(values.min()) * 0.95, float(values.max() - values.min()) + 1e-3, 0.5)
    bounds = ([0.0, 0.0, 0.0], [float(values.min()), np.inf, 5.0])
    parameters, _ = curve_fit(power_law, sizes, values, p0=initial, bounds=bounds, maxfev=MAX_FIT_EVALUATIONS)
    return LearningCurve(*(float(value) for value in parameters))


def fold_table(curve: LearningCurve, fold_candidates: tuple[int, ...], n_available: int) -> pd.DataFrame:
    """Predicted loss and loss increase of each candidate number of folds.

    Args:
        curve: Fitted learning curve.
        fold_candidates: Numbers of folds k.
        n_available: Records available for cross-validation (training plus validation).

    Returns:
        Table indexed by k with training records per fold, predicted loss, loss increase compared
        with training on all ``n_available`` records, and relative cost (k trainings).
    """
    folds = np.asarray(sorted(fold_candidates))
    training_records = n_available * (folds - 1) / folds
    predicted = curve.loss(training_records)
    return pd.DataFrame(
        {
            "training_records": training_records.round().astype(int),
            "predicted_loss": predicted,
            "loss_increase": predicted - curve.loss(n_available),
            "relative_cost": folds,
        },
        index=pd.Index(folds, name="folds"),
    )


def suggest_folds(table: pd.DataFrame, tolerance: float) -> int:
    """Smallest number of folds whose loss increase is within the tolerance.

    Args:
        table: Output of ``fold_table``.
        tolerance: Largest acceptable loss increase.

    Returns:
        Suggested k (the largest candidate if none is within the tolerance).
    """
    within = table.index[table["loss_increase"] <= tolerance]
    return int(within.min()) if len(within) else int(table.index.max())
