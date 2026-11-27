"""Classification and calibration metrics on probabilities of the bad class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

DECISION_THRESHOLD = 0.5
CLASS_LABELS: tuple[int, int] = (0, 1)


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


def subset_metrics(
    probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    n_bins: int,
) -> dict[str, float]:
    """Classification metrics of a subset of the records.

    AUC and balanced accuracy use the raw probabilities, the calibration error the prior-corrected ones
    (as in the evaluation notebook).

    Args:
        probabilities: Raw probabilities of the bad class.
        corrected_probabilities: Prior-corrected probabilities of the bad class.
        labels: Labels (1 = bad).
        selected: Boolean mask of the records of the subset.
        n_bins: Bins of the calibration error.

    Returns:
        Dictionary with ``records``, ``bad_rate``, ``auc``, ``balanced_accuracy`` and ``ece``
        (NaN AUC when the subset has a single class).

    Raises:
        ValueError: If the subset is empty.
    """
    if not selected.any():
        raise ValueError("The subset holds no record.")
    metrics = classification_metrics(probabilities[selected], labels[selected], n_bins)
    metrics["ece"] = classification_metrics(corrected_probabilities[selected], labels[selected], n_bins)["ece"]
    return {"records": int(selected.sum()), "bad_rate": float(labels[selected].mean()), **metrics}


def confusion_counts(probabilities: np.ndarray, labels: np.ndarray, threshold: float = DECISION_THRESHOLD) -> np.ndarray:
    """Confusion matrix with the bad class as positive.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        threshold: Records with probability at or above it are predicted bad.

    Returns:
        2 x 2 integer array: rows true class (good, bad), columns predicted class (good, bad),
        that is ``[[TN, FP], [FN, TP]]``.
    """
    return confusion_matrix(np.asarray(labels).astype(int), (probabilities >= threshold).astype(int), labels=list(CLASS_LABELS))


def confusion_summary(counts: np.ndarray) -> dict[str, float]:
    """Counts and rates of a confusion matrix (bad class positive).

    Args:
        counts: Output of ``confusion_counts``.

    Returns:
        Dictionary with ``tp``, ``tn``, ``fp``, ``fn``, ``recall_bad`` (TP / (TP + FN)),
        ``specificity`` (TN / (TN + FP)) and ``precision_bad`` (TP / (TP + FP); NaN if nothing is predicted bad).
    """
    (tn, fp), (fn, tp) = counts
    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "recall_bad": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision_bad": tp / (tp + fp) if tp + fp else float("nan"),
    }


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve of the bad class (its baseline is the share of bad records).

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).

    Returns:
        Average precision.
    """
    return float(average_precision_score(labels, probabilities))
