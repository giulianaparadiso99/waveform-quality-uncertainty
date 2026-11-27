"""How much the zero padding inside the analysis window tells about the class.

The window statistics stored by the preprocessing give, for every record and
component, the seconds of the window without data (``missing_in_window_s_<c>``:
zero padding and, rarely, internal gaps). From them:
- the share of the window without data, and its separation of the classes when used
  alone as a score (the information the network could take from the padding);
- the classification metrics of the network on groups of records with increasing
  padding (does the network do better where it can rely on padding?).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from wqu.training.metrics import subset_metrics

MISSING_COLUMN_FORMAT = "missing_in_window_s_{}"
NO_PADDING_GROUP = "none"


def window_padding_share(records: pd.DataFrame, components: Sequence[str], window_length_s: float) -> pd.Series:
    """Share of the window without data, mean over components.

    Args:
        records: Feature records with the ``missing_in_window_s_<component>`` columns.
        components: Component names.
        window_length_s: Length of the analysis window.

    Returns:
        Series in [0, 1] aligned with ``records``.
    """
    columns = [MISSING_COLUMN_FORMAT.format(component) for component in components]
    return (records[columns].mean(axis=1) / window_length_s).clip(0.0, 1.0).rename("window_padding_share")


def padding_groups(shares: pd.Series, edges: Sequence[float]) -> pd.Series:
    """Group of each record by its share of padding: none, then intervals between the edges.

    Args:
        shares: Output of ``window_padding_share``.
        edges: Increasing upper limits of the groups with padding (e.g. 0.1, 0.3); the last
            group collects the larger shares.

    Returns:
        Ordered categorical series of group labels.
    """
    limits = [0.0, *edges, 1.0]
    labels = [NO_PADDING_GROUP] + [f"({low:.0%}, {high:.0%}]" for low, high in zip(limits[:-1], limits[1:])]
    group = pd.cut(shares, bins=[-np.inf, *limits], labels=labels, right=True)
    return group.astype(pd.CategoricalDtype(labels, ordered=True)).rename("padding_group")


def padding_by_class(shares: pd.Series, labels: np.ndarray, class_names: dict[int, str]) -> pd.DataFrame:
    """Share of records with padding and padding share among them, per class.

    Args:
        shares: Output of ``window_padding_share``.
        labels: Labels (1 = bad).
        class_names: Name of each label.

    Returns:
        Table with one row per class.
    """
    frame = pd.DataFrame({"share": shares.to_numpy(), "label": labels})
    rows = {}
    for label, group in frame.groupby("label"):
        padded = group.loc[group["share"] > 0, "share"]
        rows[class_names[int(label)]] = {
            "records": len(group),
            "with_padding": float((group["share"] > 0).mean()),
            "share_median_if_padded": float(padded.median()) if len(padded) else float("nan"),
            "share_mean": float(group["share"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def padding_separation(shares: pd.Series, labels: np.ndarray) -> dict[str, float]:
    """AUC of the padding share as a score of the bad class, and the corresponding separation.

    Args:
        shares: Output of ``window_padding_share``.
        labels: Labels (1 = bad).

    Returns:
        Dictionary with ``auc`` and ``separation`` (max(AUC, 1 - AUC)).
    """
    auc = float(roc_auc_score(labels, shares.to_numpy()))
    return {"auc": auc, "separation": max(auc, 1.0 - auc)}


def metrics_by_group(
    probabilities: np.ndarray,
    corrected_probabilities: np.ndarray,
    labels: np.ndarray,
    groups: pd.Series,
    n_bins: int,
) -> pd.DataFrame:
    """Classification metrics of the network in each padding group.

    AUC and balanced accuracy use the raw probabilities, the calibration error the
    prior-corrected ones (as in the evaluation notebook).

    Args:
        probabilities: Raw probabilities of the bad class.
        corrected_probabilities: Prior-corrected probabilities of the bad class.
        labels: Labels (1 = bad).
        groups: Output of ``padding_groups``, aligned with the probabilities.
        n_bins: Bins of the calibration error.

    Returns:
        Table with one row per group: records, bad rate, AUC, balanced accuracy and ECE
        (AUC is NaN when a group has a single class).
    """
    rows = {}
    for group in groups.cat.categories:
        selected = (groups == group).to_numpy()
        if selected.any():
            rows[group] = subset_metrics(probabilities, corrected_probabilities, labels, selected, n_bins)
    return pd.DataFrame.from_dict(rows, orient="index")
