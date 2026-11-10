"""Classification and calibration metrics on probabilities of the bad class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

DECISION_THRESHOLD = 0.5


def reliability_table(probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Records, mean probability and observed bad rate in equal-width probability bins.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        n_bins: Number of bins on [0, 1].

    Returns:
        Table with one row per bin: ``bin_lower``, ``bin_upper``, ``records``,
        ``mean_probability`` and ``observed_bad_rate`` (NaN in empty bins).
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    grouped = pd.DataFrame({"bin": bins, "probability": probabilities, "label": labels}).groupby("bin")
    table = pd.DataFrame({"bin_lower": edges[:-1], "bin_upper": edges[1:]})
    table["records"] = grouped.size().reindex(table.index, fill_value=0).to_numpy()
    table["mean_probability"] = grouped["probability"].mean().reindex(table.index).to_numpy()
    table["observed_bad_rate"] = grouped["label"].mean().reindex(table.index).to_numpy()
    return table


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Expected calibration error with equal-width bins of the bad-class probability.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        n_bins: Number of bins on [0, 1].

    Returns:
        Weighted mean over bins of |share of bad records - mean probability|.
    """
    table = reliability_table(probabilities, labels, n_bins)
    filled = table[table["records"] > 0]
    gaps = (filled["observed_bad_rate"] - filled["mean_probability"]).abs()
    return float((filled["records"] * gaps).sum() / filled["records"].sum())


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
