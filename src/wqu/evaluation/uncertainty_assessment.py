"""Does the uncertainty point at the wrong predictions?

Each uncertainty measure is paired with the prediction it refers to: the entropy
of the deterministic pass with the deterministic prediction, the MC dropout
measures with the mean of the MC passes. A prediction is wrong when the
probability of the bad class, thresholded at 0.5, disagrees with the label (the
same decision used for the balanced accuracy).

Three views:
- error rate in groups of equal size ordered by uncertainty;
- AUC of the uncertainty as a detector of wrong predictions (0.5 = useless);
- rejection curves: accuracy and balanced accuracy on the records kept after
  discarding the most uncertain ones, against random rejection and an oracle that
  discards the wrong predictions first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from wqu.training.metrics import DECISION_THRESHOLD

MEASURE_PREDICTIONS: Mapping[str, str] = {
    "deterministic_entropy": "probability_deterministic",
    "probability_std": "probability_mean",
    "predictive_entropy": "probability_mean",
    "mutual_information": "probability_mean",
}
REFERENCE_PREDICTION = "probability_mean"
RANDOM_METHOD = "random"
ORACLE_METHOD = "oracle"


def predicted_labels(probabilities: np.ndarray) -> np.ndarray:
    """Predicted class (1 = bad) at the decision threshold.

    Args:
        probabilities: Probabilities of the bad class.

    Returns:
        Integer array of predicted labels.
    """
    return (np.asarray(probabilities) >= DECISION_THRESHOLD).astype(int)


def prediction_errors(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Whether each prediction is wrong.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).

    Returns:
        Boolean array, true for wrong predictions.
    """
    return predicted_labels(probabilities) != np.asarray(labels).astype(int)


def retained_fractions(min_fraction: float, step: float) -> np.ndarray:
    """Fractions of records kept, from 1 down to ``min_fraction``.

    Args:
        min_fraction: Smallest fraction.
        step: Step between fractions.

    Returns:
        Decreasing array starting at 1.
    """
    n_steps = int(np.floor((1.0 - min_fraction) / step + 1e-9))
    return np.round(1.0 - step * np.arange(n_steps + 1), 6)


def error_by_uncertainty(uncertainty: np.ndarray, errors: np.ndarray, n_groups: int) -> pd.DataFrame:
    """Error rate in groups of equal size ordered by increasing uncertainty.

    Ties are broken by position, so that the groups have equal size even when many
    records share the same uncertainty.

    Args:
        uncertainty: Uncertainty of each record.
        errors: Whether each prediction is wrong.
        n_groups: Number of groups.

    Returns:
        Table with one row per group (1 = least uncertain): ``records``,
        ``uncertainty_min``, ``uncertainty_max``, ``uncertainty_mean`` and ``error_rate``.
    """
    ranks = pd.Series(uncertainty).rank(method="first")
    groups = pd.qcut(ranks, n_groups, labels=False) + 1
    frame = pd.DataFrame({"group": groups.to_numpy(), "uncertainty": uncertainty, "error": errors})
    table = frame.groupby("group").agg(
        records=("error", "size"),
        uncertainty_min=("uncertainty", "min"),
        uncertainty_max=("uncertainty", "max"),
        uncertainty_mean=("uncertainty", "mean"),
        error_rate=("error", "mean"),
    )
    return table


def error_detection_auc(uncertainty: np.ndarray, errors: np.ndarray) -> float:
    """AUC of the uncertainty as a score that separates wrong from right predictions.

    Args:
        uncertainty: Uncertainty of each record.
        errors: Whether each prediction is wrong.

    Returns:
        AUC (NaN if all predictions are right or all wrong).
    """
    if len(np.unique(errors)) < 2:
        return float("nan")
    return float(roc_auc_score(errors, uncertainty))


def retained_metrics(
    rejection_order: np.ndarray, labels: np.ndarray, predicted: np.ndarray, fractions: np.ndarray
) -> pd.DataFrame:
    """Accuracy and balanced accuracy on the records kept at each retained fraction.

    Args:
        rejection_order: Record positions from the first to keep to the first to discard.
        labels: Labels (1 = bad).
        predicted: Predicted labels.
        fractions: Fractions of records kept.

    Returns:
        Table with ``retained_fraction``, ``accuracy`` and ``balanced_accuracy``.
    """
    rows = []
    for fraction in fractions:
        kept = rejection_order[: max(1, int(round(fraction * len(rejection_order))))]
        kept_labels, kept_predicted = labels[kept], predicted[kept]
        balanced = (
            float(balanced_accuracy_score(kept_labels, kept_predicted))
            if len(np.unique(kept_labels)) == 2
            else float("nan")
        )
        rows.append(
            {
                "retained_fraction": float(fraction),
                "accuracy": float((kept_labels == kept_predicted).mean()),
                "balanced_accuracy": balanced,
            }
        )
    return pd.DataFrame(rows)


def rejection_curve(
    uncertainty: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, fractions: np.ndarray
) -> pd.DataFrame:
    """Metrics on the records kept after discarding the most uncertain ones.

    Args:
        uncertainty: Uncertainty of each record.
        labels: Labels (1 = bad).
        probabilities: Probabilities of the prediction the uncertainty refers to.
        fractions: Fractions of records kept.

    Returns:
        Output of ``retained_metrics``.
    """
    order = np.argsort(uncertainty, kind="stable")
    return retained_metrics(order, labels, predicted_labels(probabilities), fractions)


def random_rejection_curve(
    labels: np.ndarray, probabilities: np.ndarray, fractions: np.ndarray, repeats: int, seed: int
) -> pd.DataFrame:
    """Mean metrics over random orders of rejection (baseline without uncertainty).

    Args:
        labels: Labels (1 = bad).
        probabilities: Probabilities of the bad class.
        fractions: Fractions of records kept.
        repeats: Number of random orders.
        seed: Random seed.

    Returns:
        Output of ``retained_metrics`` averaged over the orders.
    """
    generator = np.random.default_rng(seed)
    predicted = predicted_labels(probabilities)
    curves = [
        retained_metrics(generator.permutation(len(labels)), labels, predicted, fractions) for _ in range(repeats)
    ]
    return pd.concat(curves).groupby("retained_fraction", sort=False).mean().reset_index()


def oracle_rejection_curve(labels: np.ndarray, probabilities: np.ndarray, fractions: np.ndarray) -> pd.DataFrame:
    """Metrics when the wrong predictions are discarded first (best possible rejection).

    Args:
        labels: Labels (1 = bad).
        probabilities: Probabilities of the bad class.
        fractions: Fractions of records kept.

    Returns:
        Output of ``retained_metrics``.
    """
    errors = prediction_errors(probabilities, labels)
    return retained_metrics(np.argsort(errors, kind="stable"), labels, predicted_labels(probabilities), fractions)


def error_rate_tables(
    predictions: pd.DataFrame, label_column: str, measures: Sequence[str], n_groups: int
) -> dict[str, pd.DataFrame]:
    """Error rate by uncertainty group for each measure.

    Args:
        predictions: Table with the labels, the probabilities and the uncertainty measures.
        label_column: Name of the label column.
        measures: Uncertainty measures (keys of ``MEASURE_PREDICTIONS``).
        n_groups: Number of groups.

    Returns:
        Mapping from measure to the output of ``error_by_uncertainty``.
    """
    labels = predictions[label_column].to_numpy()
    return {
        measure: error_by_uncertainty(
            predictions[measure].to_numpy(),
            prediction_errors(predictions[MEASURE_PREDICTIONS[measure]].to_numpy(), labels),
            n_groups,
        )
        for measure in measures
    }


def error_detection_table(predictions: pd.DataFrame, label_column: str, measures: Sequence[str]) -> pd.DataFrame:
    """Error rate of the prediction and error-detection AUC of each measure.

    Args:
        predictions: Table with the labels, the probabilities and the uncertainty measures.
        label_column: Name of the label column.
        measures: Uncertainty measures.

    Returns:
        Table indexed by measure with ``prediction``, ``error_rate`` and ``error_detection_auc``.
    """
    labels = predictions[label_column].to_numpy()
    rows = {}
    for measure in measures:
        errors = prediction_errors(predictions[MEASURE_PREDICTIONS[measure]].to_numpy(), labels)
        rows[measure] = {
            "prediction": MEASURE_PREDICTIONS[measure],
            "error_rate": float(errors.mean()),
            "error_detection_auc": error_detection_auc(predictions[measure].to_numpy(), errors),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def rejection_curves(
    predictions: pd.DataFrame,
    label_column: str,
    measures: Sequence[str],
    fractions: np.ndarray,
    random_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Rejection curves of every measure, of random rejection and of the oracle.

    The random and oracle curves use the MC dropout mean prediction.

    Args:
        predictions: Table with the labels, the probabilities and the uncertainty measures.
        label_column: Name of the label column.
        measures: Uncertainty measures.
        fractions: Fractions of records kept.
        random_repeats: Random orders averaged for the random baseline.
        seed: Seed of the random orders.

    Returns:
        Long table with ``method``, ``retained_fraction``, ``accuracy`` and ``balanced_accuracy``.
    """
    labels = predictions[label_column].to_numpy().astype(int)
    reference = predictions[REFERENCE_PREDICTION].to_numpy()
    curves = {
        measure: rejection_curve(
            predictions[measure].to_numpy(), labels, predictions[MEASURE_PREDICTIONS[measure]].to_numpy(), fractions
        )
        for measure in measures
    }
    curves[RANDOM_METHOD] = random_rejection_curve(labels, reference, fractions, random_repeats, seed)
    curves[ORACLE_METHOD] = oracle_rejection_curve(labels, reference, fractions)
    return pd.concat([curve.assign(method=method) for method, curve in curves.items()], ignore_index=True)


def rejection_summary(curves: pd.DataFrame, fractions: Sequence[float], metric: str) -> pd.DataFrame:
    """One metric of the rejection curves at selected retained fractions.

    Args:
        curves: Output of ``rejection_curves``.
        fractions: Retained fractions to report.
        metric: ``accuracy`` or ``balanced_accuracy``.

    Returns:
        Table with methods as rows and retained fractions as columns, from 1 (all records)
        down to the smallest reported fraction.
    """
    table = curves.pivot(index="method", columns="retained_fraction", values=metric)
    reported = (1.0, *sorted(set(fractions) - {1.0}, reverse=True))
    columns = [table.columns[np.argmin(np.abs(table.columns.to_numpy() - fraction))] for fraction in reported]
    return table[columns].reindex(curves["method"].unique())
