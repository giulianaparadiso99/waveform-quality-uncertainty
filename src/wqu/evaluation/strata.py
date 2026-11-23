"""Performance of the network in strata of hypocentral distance and magnitude.

The network sees only the time-frequency images, but distance and magnitude shape them
(amplitude, frequency content, signal-to-noise ratio, duration), and in ESM25 the share of
bad records grows with distance. Two checks tell whether the network separates the classes
beyond what distance and magnitude explain:
- the classification metrics within strata of distance, of magnitude and of both: if the
  network relied only on distance, the AUC within a distance stratum would drop towards 0.5;
- the mean predicted probability and the error rate of each class along the distance strata,
  with the rank correlation between probability and distance within each class: a probability
  of bad that grows with distance among the good records means that distance pushes the
  network towards bad beyond the true class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from wqu.evaluation.uncertainty_assessment import predicted_labels

DISTANCE_COLUMN = "path_hyp_dist_km"
MAGNITUDE_COLUMN = "preferred_magnitude"


def interval_groups(values: pd.Series, edges: Sequence[float], label_format: str = "{:g}") -> pd.Series:
    """Ordered stratum of every value, between consecutive edges.

    Strata are closed on the left: ``< e0``, ``[e0, e1)``, ..., ``>= en``.

    Args:
        values: Values to stratify (missing values get no stratum).
        edges: Increasing interior edges.
        label_format: Format of the edges in the labels.

    Returns:
        Ordered categorical series aligned with ``values``.

    Raises:
        ValueError: If the edges are empty or not strictly increasing.
    """
    limits = np.asarray(edges, dtype=float)
    if limits.size == 0 or np.any(np.diff(limits) <= 0):
        raise ValueError("The edges must be a non-empty, strictly increasing sequence.")
    texts = [label_format.format(edge) for edge in limits]
    labels = [f"< {texts[0]}"]
    labels += [f"[{low}, {high})" for low, high in zip(texts[:-1], texts[1:])]
    labels.append(f">= {texts[-1]}")
    groups = pd.cut(values, bins=[-np.inf, *limits, np.inf], labels=labels, right=False)
    return groups.astype(pd.CategoricalDtype(labels, ordered=True))


def auc_grid(
    probabilities: np.ndarray,
    labels: np.ndarray,
    row_groups: pd.Series,
    column_groups: pd.Series,
    min_class_records: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUC and number of records in every cell of a two-way stratification.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        row_groups: Ordered categorical stratum of each record for the rows.
        column_groups: Ordered categorical stratum of each record for the columns.
        min_class_records: Records of each class needed for an AUC (NaN otherwise).

    Returns:
        Tuple (AUC table, records table), rows and columns in the order of the categories.
    """
    rows, columns = row_groups.cat.categories, column_groups.cat.categories
    auc = pd.DataFrame(np.nan, index=rows, columns=columns)
    counts = pd.DataFrame(0, index=rows, columns=columns)
    row_values, column_values = row_groups.to_numpy(), column_groups.to_numpy()
    for row in rows:
        for column in columns:
            selected = (row_values == row) & (column_values == column)
            counts.loc[row, column] = int(selected.sum())
            n_bad = int(labels[selected].sum())
            if min(n_bad, int(selected.sum()) - n_bad) >= min_class_records:
                auc.loc[row, column] = roc_auc_score(labels[selected], probabilities[selected])
    return auc, counts


def class_trends(
    probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: pd.Series,
    class_names: Mapping[int, str],
) -> pd.DataFrame:
    """Observed bad rate, mean probability and error rate of each class in every stratum.

    Args:
        probabilities: Raw probabilities of the bad class (for the errors, threshold 0.5).
        corrected_probabilities: Prior-corrected probabilities (for the mean probability).
        labels: Labels (1 = bad).
        groups: Ordered categorical stratum of each record.
        class_names: Name of each label.

    Returns:
        Table with one row per stratum: records, observed bad rate, mean corrected probability
        of all records and, per class, records, mean corrected probability and error rate
        (good records predicted bad, bad records predicted good).
    """
    wrong = predicted_labels(probabilities) != labels
    stratum_values = groups.to_numpy()
    rows = {}
    for stratum in groups.cat.categories:
        selected = stratum_values == stratum
        row = {
            "records": int(selected.sum()),
            "observed_bad_rate": float(labels[selected].mean()) if selected.any() else np.nan,
            "mean_probability": float(corrected_probabilities[selected].mean()) if selected.any() else np.nan,
        }
        for label, name in class_names.items():
            in_class = selected & (labels == label)
            row[f"records_{name}"] = int(in_class.sum())
            row[f"mean_probability_{name}"] = float(corrected_probabilities[in_class].mean()) if in_class.any() else np.nan
            row[f"error_rate_{name}"] = float(wrong[in_class].mean()) if in_class.any() else np.nan
        rows[stratum] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def within_class_rank_correlation(
    probabilities: np.ndarray, labels: np.ndarray, values: np.ndarray, class_names: Mapping[int, str]
) -> pd.DataFrame:
    """Spearman correlation between a variable and the predicted probability within each class.

    Args:
        probabilities: Probabilities of the bad class.
        labels: Labels (1 = bad).
        values: Variable of each record (e.g. distance); records with missing values are skipped.
        class_names: Name of each label.

    Returns:
        Table with one row per class: records, Spearman correlation and its p-value.
    """
    valid = np.isfinite(values)
    rows = {}
    for label, name in class_names.items():
        in_class = valid & (labels == label)
        result = spearmanr(values[in_class], probabilities[in_class])
        rows[name] = {"records": int(in_class.sum()), "spearman": float(result.statistic),
                      "p_value": float(result.pvalue)}
    return pd.DataFrame.from_dict(rows, orient="index")
