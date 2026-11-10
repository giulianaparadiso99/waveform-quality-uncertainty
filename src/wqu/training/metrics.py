"""Classification and calibration metrics on probabilities of the bad class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

DECISION_THRESHOLD = 0.5


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Expected calibration error with equal-width bins of the bad-class probability.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        n_bins: Number of bins on [0, 1].

    Returns:
        Weighted mean over bins of |share of bad records - mean probability|.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for current in range(n_bins):
        inside = bins == current
        if inside.any():
            error += inside.mean() * abs(labels[inside].mean() - probabilities[inside].mean())
    return float(error)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> dict[str, float]:
    """AUC, balanced accuracy at threshold 0.5 and expected calibration error.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        n_bins: Bins of the calibration error.

    Returns:
        Dictionary with ``auc``, ``balanced_accuracy`` and ``ece`` (NaN AUC when one class is absent).
    """
    auc = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return {
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(labels, probabilities >= DECISION_THRESHOLD)),
        "ece": expected_calibration_error(probabilities, labels, n_bins),
    }
