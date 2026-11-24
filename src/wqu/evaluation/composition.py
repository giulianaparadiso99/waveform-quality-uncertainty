"""Metrics of the validation predictions on subsets of different composition.

Some metrics depend on the share of bad records in the evaluated set and others do not: the
AUC and the balanced accuracy do not (they compare the two classes separately), while the
accuracy, the precision on the bad class and the average precision grow with the share of bad
records. Recomputing the metrics on subsets of the validation set, for example with balanced
classes or with a minimum magnitude as in another study, shows how much of a difference in
reported metrics can come from the composition of the evaluated data rather than from the model.

Balanced subsets are drawn by keeping all the records of the smaller class and as many records,
drawn at random without replacement, of the larger one; the metrics are averaged over several
draws.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from wqu.evaluation.uncertainty_assessment import predicted_labels

BALANCED_COLUMN = "balanced"


def balanced_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Positions of a class-balanced random subset: the smaller class whole, the larger one subsampled.

    Args:
        labels: Labels (1 = bad).
        rng: Random generator.

    Returns:
        Sorted positions into ``labels``.

    Raises:
        ValueError: If one of the classes is absent.
    """
    good, bad = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    if not len(good) or not len(bad):
        raise ValueError("Both classes are needed for a balanced subset.")
    size = min(len(good), len(bad))
    drawn = [rng.choice(positions, size, replace=False) if len(positions) > size else positions
             for positions in (good, bad)]
    return np.sort(np.concatenate(drawn))


def threshold_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Metrics of raw probabilities of the bad class, with the decision threshold 0.5.

    Args:
        probabilities: Raw probabilities of the bad class.
        labels: Labels (1 = bad).

    Returns:
        Records, share of bad records, AUC, average precision, accuracy, balanced accuracy,
        precision and recall on the bad class.
    """
    predicted = predicted_labels(probabilities)
    return {
        "records": float(len(labels)),
        "bad_rate": float(labels.mean()),
        "auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "precision_bad": float(precision_score(labels, predicted, zero_division=0)),
        "recall_bad": float(recall_score(labels, predicted, zero_division=0)),
    }


def balanced_metrics(probabilities: np.ndarray, labels: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    """Metrics averaged over several class-balanced random subsets.

    Args:
        probabilities: Raw probabilities of the bad class.
        labels: Labels (1 = bad).
        repeats: Number of balanced subsets drawn.
        seed: Seed of the draws.

    Returns:
        Mean of ``threshold_metrics`` over the subsets, with ``auc_std`` (standard deviation of
        the AUC across the draws).
    """
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        positions = balanced_indices(labels, rng)
        draws.append(threshold_metrics(probabilities[positions], labels[positions]))
    table = pd.DataFrame(draws)
    return {**table.mean().to_dict(), "auc_std": float(table["auc"].std(ddof=1)) if repeats > 1 else 0.0}


def composition_table(
    probabilities: np.ndarray,
    labels: np.ndarray,
    selections: Mapping[str, np.ndarray],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Metrics of each selection of records, as it is and with balanced classes.

    Args:
        probabilities: Raw probabilities of the bad class.
        labels: Labels (1 = bad).
        selections: Name of each selection and its boolean mask over the records.
        repeats: Balanced subsets drawn per selection.
        seed: Seed of the draws.

    Returns:
        Table indexed by selection and ``balanced`` (False, True) with the metrics of
        ``threshold_metrics`` (means over the draws for the balanced rows) and ``auc_std``.
    """
    rows = {}
    for name, mask in selections.items():
        selected_probabilities, selected_labels = probabilities[mask], labels[mask]
        rows[(name, False)] = {**threshold_metrics(selected_probabilities, selected_labels), "auc_std": 0.0}
        rows[(name, True)] = balanced_metrics(selected_probabilities, selected_labels, repeats, seed)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["selection", BALANCED_COLUMN])
    return table
