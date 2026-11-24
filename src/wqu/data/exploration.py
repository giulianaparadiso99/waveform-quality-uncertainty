"""Exploratory analysis of the ESM25 metadata.

All functions are pure: they take tables and parameters and return tables,
leaving plotting to ``wqu.visualization``. The label convention is
1 = bad-quality record, 0 = good-quality record, so "bad rate" is the mean label.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from wqu.config import FeatureGroup

MISSING_CATEGORY = "missing"
OTHER_CATEGORY = "other"
# Significant digits of the bin edges shown in labels.
BIN_LABEL_DIGITS = 4
_NUMBER = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
NUMERIC_LEVEL_PATTERN = re.compile(rf"^({_NUMBER})(?: to {_NUMBER})?$")


def compare_columns(metadata_by_subset: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabulate which columns are present in which subset.

    Args:
        metadata_by_subset: Metadata per subset.

    Returns:
        Boolean table (one row per column, one column per subset) with an extra
        ``n_subsets`` column, sorted so that shared columns come first.
    """
    all_columns = pd.Index([])
    for frame in metadata_by_subset.values():
        all_columns = all_columns.union(frame.columns, sort=False)
    presence = pd.DataFrame(
        {name: all_columns.isin(frame.columns) for name, frame in metadata_by_subset.items()},
        index=all_columns,
    )
    presence["n_subsets"] = presence.sum(axis=1)
    return presence.sort_values("n_subsets", ascending=False, kind="stable")


def assert_features_shared(metadata_by_subset: Mapping[str, pd.DataFrame], features: Sequence[str]) -> None:
    """Raise a ValueError if a feature is missing from any subset.

    A column present in one class only would separate the classes through its
    missing values, so it cannot be used to measure confounding.

    Args:
        metadata_by_subset: Metadata per subset.
        features: Feature columns to check.
    """
    presence = compare_columns(metadata_by_subset)
    absent = [feature for feature in features if feature not in presence.index]
    not_shared = [
        feature for feature in features
        if feature in presence.index and presence.loc[feature, "n_subsets"] < len(metadata_by_subset)
    ]
    if absent or not_shared:
        raise ValueError(f"Features absent from all subsets: {absent}; present in some subsets only: {not_shared}.")


def constant_columns(metadata: pd.DataFrame, columns: Sequence[str] | None = None) -> list[str]:
    """Return the columns that take a single value (missing counted as a value).

    Args:
        metadata: Input table.
        columns: Columns to check; all columns if None.

    Returns:
        Names of constant columns.
    """
    selected = list(columns) if columns is not None else list(metadata.columns)
    return [column for column in selected if metadata[column].nunique(dropna=False) <= 1]


def value_counts_by_label(metadata: pd.DataFrame, column: str, label_column: str) -> pd.DataFrame:
    """Cross-tabulate the values of a column against the label.

    Args:
        metadata: Combined metadata.
        column: Column to tabulate (missing values shown as their own row).
        label_column: Label column.

    Returns:
        Count table (values x labels).
    """
    return pd.crosstab(metadata[column].astype("object").fillna(MISSING_CATEGORY), metadata[label_column])


def missing_fraction_by_label(metadata: pd.DataFrame, columns: Sequence[str], label_column: str) -> pd.DataFrame:
    """Fraction of missing values per column and label.

    Args:
        metadata: Combined metadata.
        columns: Columns to check.
        label_column: Label column.

    Returns:
        Table (columns x labels) of missing fractions.
    """
    return metadata[list(columns)].isna().groupby(metadata[label_column]).mean().T


def numeric_summary_by_label(metadata: pd.DataFrame, columns: Sequence[str], label_column: str) -> pd.DataFrame:
    """Descriptive statistics of numeric columns for each label.

    Args:
        metadata: Combined metadata.
        columns: Numeric columns.
        label_column: Label column.

    Returns:
        Table indexed by (column, label) with count, mean, std and quantiles.
    """
    summaries = {
        column: pd.to_numeric(metadata[column], errors="coerce").groupby(metadata[label_column]).describe()
        for column in columns
    }
    return pd.concat(summaries, names=["column", label_column])


def to_categories(values: pd.Series, top_n: int) -> pd.Series:
    """Map a categorical column to its most frequent levels.

    Levels outside the ``top_n`` most frequent become ``"other"`` and missing
    values become ``"missing"``.

    Args:
        values: Categorical values.
        top_n: Number of levels kept.

    Returns:
        Series of strings.
    """
    as_text = values.astype("object").where(values.notna(), MISSING_CATEGORY).astype(str)
    kept = as_text[as_text != MISSING_CATEGORY].value_counts().index[:top_n]
    return as_text.where(as_text.isin(kept) | (as_text == MISSING_CATEGORY), OTHER_CATEGORY)


def to_quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Map a numeric column to quantile bins labelled by their interval.

    Args:
        values: Numeric values.
        n_bins: Requested number of bins (fewer if quantiles coincide).

    Returns:
        Series of labels such as ``"4.1 to 4.31"``, ``"missing"`` for missing values.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    codes, edges = pd.qcut(numeric, q=n_bins, duplicates="drop", labels=False, retbins=True)
    labels = [f"{low:.{BIN_LABEL_DIGITS}g} to {high:.{BIN_LABEL_DIGITS}g}" for low, high in zip(edges[:-1], edges[1:])]
    binned = codes.map(lambda code: labels[int(code)] if pd.notna(code) else MISSING_CATEGORY)
    return binned.astype(str)


def to_strata(values: pd.Series, n_bins: int, top_n: int, max_discrete_levels: int) -> pd.Series:
    """Convert a column into discrete strata.

    Numeric columns with more than ``max_discrete_levels`` distinct values are
    binned by quantiles; other columns keep their ``top_n`` most frequent levels.

    Args:
        values: Column to convert.
        n_bins: Number of quantile bins for numeric columns.
        top_n: Number of levels kept for discrete columns.
        max_discrete_levels: Largest number of distinct numeric values treated as categories.

    Returns:
        Series of stratum labels.
    """
    is_numeric = pd.api.types.is_numeric_dtype(values)
    if is_numeric and values.nunique(dropna=True) > max_discrete_levels:
        return to_quantile_bins(values, n_bins)
    return to_categories(values, top_n)


def level_sort_key(level: object) -> tuple[int, float, str]:
    """Sort key placing numeric levels (or intervals) first in numeric order.

    Levels such as ``"80.0"`` or ``"4.2 to 4.6"`` are ordered by their first
    number; other levels follow alphabetically, with ``"other"`` and
    ``"missing"`` last.

    Args:
        level: Level label.

    Returns:
        Tuple usable as sort key.
    """
    text = str(level)
    if text in (OTHER_CATEGORY, MISSING_CATEGORY):
        return (2, 0.0, text)
    match = NUMERIC_LEVEL_PATTERN.match(text)
    if match is not None:
        return (0, float(match.group(1)), text)
    return (1, 0.0, text)


def sort_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Reorder the rows and columns of a level table with ``level_sort_key``.

    Args:
        table: Table indexed by levels on both axes.

    Returns:
        Reordered table.
    """
    row_order = sorted(table.index, key=level_sort_key)
    column_order = sorted(table.columns, key=level_sort_key)
    return table.loc[row_order, column_order]


def categorical_distribution_by_label(
    metadata: pd.DataFrame, column: str, label_column: str, top_n: int
) -> pd.DataFrame:
    """Number of records and bad rate for the most frequent levels of a column.

    Numeric columns are treated as categories, so this function suits discrete
    numeric variables such as the original sampling rate.

    Args:
        metadata: Combined metadata.
        column: Categorical or discrete numeric column.
        label_column: Label column (1 = bad).
        top_n: Number of levels kept; the rest are grouped as ``"other"``.

    Returns:
        Table indexed by level with ``n_records``, ``n_bad`` and ``bad_rate``,
        sorted by number of records.
    """
    levels = to_categories(metadata[column], top_n)
    grouped = metadata[label_column].groupby(levels)
    table = pd.DataFrame({"n_records": grouped.size(), "n_bad": grouped.sum()})
    table["bad_rate"] = table["n_bad"] / table["n_records"]
    table.index.name = column
    return table.sort_values("n_records", ascending=False)


def event_overlap_summary(metadata: pd.DataFrame, event_column: str, label_column: str) -> pd.DataFrame:
    """Count events by the classes of their records.

    Events with both good and bad records must stay in a single split, so this
    table shows how much of the dataset is affected by grouping.

    Args:
        metadata: Combined metadata.
        event_column: Event identifier column.
        label_column: Label column (1 = bad).

    Returns:
        Table indexed by composition (``only_good``, ``only_bad``, ``both``) with
        the number of events and records.
    """
    bad_rate_per_event = metadata.groupby(event_column, observed=True)[label_column].mean()
    composition = pd.Series(
        np.select(
            [bad_rate_per_event == 0, bad_rate_per_event == 1],
            ["only_good", "only_bad"],
            default="both",
        ),
        index=bad_rate_per_event.index,
    )
    records_per_event = metadata.groupby(event_column, observed=True).size()
    table = pd.DataFrame(
        {
            "n_events": composition.value_counts(),
            "n_records": records_per_event.groupby(composition).sum(),
        }
    ).reindex(["only_good", "only_bad", "both"]).fillna(0).astype(int)
    table.index.name = "composition"
    return table


def group_size_summary(metadata: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    """Distribution of the number of records per group.

    Args:
        metadata: Combined metadata.
        group_columns: Grouping columns (e.g. event, station, year).

    Returns:
        Table with one row per grouping column: number of groups and quantiles of
        the records per group.
    """
    rows = {}
    for column in group_columns:
        sizes = metadata.groupby(column, observed=True).size()
        rows[column] = {"n_groups": len(sizes), **sizes.describe().drop("count").to_dict()}
    return pd.DataFrame(rows).T


def numeric_univariate_auc(values: pd.Series, labels: pd.Series) -> dict[str, float]:
    """ROC AUC of a numeric variable used directly as a bad-quality score.

    Args:
        values: Numeric feature.
        labels: Labels (1 = bad).

    Returns:
        Dictionary with ``auc`` (above 0.5 when larger values indicate bad
        records), ``separation`` (max of auc and 1 - auc) and ``coverage``
        (fraction of records with a value).
    """
    numeric = pd.to_numeric(values, errors="coerce")
    available = numeric.notna()
    coverage = float(available.mean())
    if labels[available].nunique() < 2:
        return {"auc": np.nan, "separation": np.nan, "coverage": coverage}
    auc = float(roc_auc_score(labels[available], numeric[available]))
    return {"auc": auc, "separation": max(auc, 1.0 - auc), "coverage": coverage}


def out_of_fold_bad_rate(
    categories: pd.Series,
    labels: pd.Series,
    groups: pd.Series,
    n_splits: int,
    smoothing: float,
    seed: int,
) -> pd.Series:
    """Encode each category by its smoothed bad rate estimated on other folds.

    Folds are grouped (e.g. by event), so a record is never scored with
    statistics computed from records of the same group. Categories unseen in the
    training folds receive the training bad rate.

    Args:
        categories: Categorical values (missing values form their own category).
        labels: Labels (1 = bad).
        groups: Group identifiers for the folds.
        n_splits: Number of folds.
        smoothing: Pseudo-count towards the training bad rate.
        seed: Random seed of the fold assignment.

    Returns:
        Out-of-fold scores aligned with the input index.
    """
    as_text = categories.astype("object").where(categories.notna(), MISSING_CATEGORY).astype(str)
    scores = pd.Series(np.nan, index=categories.index, dtype=float)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_positions, test_positions in splitter.split(as_text, labels, groups):
        train_labels = labels.iloc[train_positions]
        prior = float(train_labels.mean())
        stats = train_labels.groupby(as_text.iloc[train_positions]).agg(["sum", "count"])
        encoding = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        test_index = categories.index[test_positions]
        scores.loc[test_index] = as_text.iloc[test_positions].map(encoding).fillna(prior).to_numpy()
    return scores


def categorical_univariate_auc(
    values: pd.Series,
    labels: pd.Series,
    groups: pd.Series,
    n_splits: int,
    smoothing: float,
    seed: int,
) -> dict[str, float]:
    """ROC AUC of a categorical variable through out-of-fold bad-rate encoding.

    Args:
        values: Categorical feature.
        labels: Labels (1 = bad).
        groups: Group identifiers for the folds.
        n_splits: Number of folds.
        smoothing: Pseudo-count of the encoding.
        seed: Random seed.

    Returns:
        Dictionary with ``auc``, ``separation``, ``coverage`` (fraction of non-missing
        values) and ``n_categories``.
    """
    scores = out_of_fold_bad_rate(values, labels, groups, n_splits, smoothing, seed)
    auc = float(roc_auc_score(labels, scores))
    return {
        "auc": auc,
        "separation": max(auc, 1.0 - auc),
        "coverage": float(values.notna().mean()),
        "n_categories": float(values.nunique(dropna=True)),
    }


def univariate_auc_table(
    metadata: pd.DataFrame,
    feature_groups: Sequence[FeatureGroup],
    label_column: str,
    group_column: str,
    n_splits: int,
    smoothing: float,
    seed: int,
) -> pd.DataFrame:
    """Univariate ROC AUC of every feature against the quality label.

    Numeric features are used directly as scores. Categorical features are
    encoded by out-of-fold bad rate with folds grouped by ``group_column``.

    Args:
        metadata: Combined metadata.
        feature_groups: Features grouped by interpretation.
        label_column: Label column (1 = bad).
        group_column: Grouping column for the folds (e.g. event identifier).
        n_splits: Number of folds.
        smoothing: Pseudo-count of the categorical encoding.
        seed: Random seed.

    Returns:
        Table with one row per feature, sorted by group and separation.
    """
    labels = metadata[label_column].astype(int)
    groups = metadata[group_column]
    rows = []
    for group in feature_groups:
        for feature in group.numeric:
            result = numeric_univariate_auc(metadata[feature], labels)
            rows.append({"group": group.name, "feature": feature, "kind": "numeric", **result})
        for feature in group.categorical:
            result = categorical_univariate_auc(metadata[feature], labels, groups, n_splits, smoothing, seed)
            rows.append({"group": group.name, "feature": feature, "kind": "categorical", **result})
    table = pd.DataFrame(rows)
    return table.sort_values(["group", "separation"], ascending=[True, False], ignore_index=True)


def stratified_bad_rate(
    metadata: pd.DataFrame,
    variable: str,
    stratum: str,
    label_column: str,
    n_bins: int,
    top_n: int,
    max_discrete_levels: int,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bad rate of the levels of a variable within the levels of a stratum.

    Used to check whether an association between a variable and the label
    persists inside each stratum (for example inside each network).

    Args:
        metadata: Combined metadata.
        variable: Variable whose levels form the rows.
        stratum: Variable whose levels form the columns.
        label_column: Label column (1 = bad).
        n_bins: Quantile bins for numeric columns with many values.
        top_n: Levels kept for discrete columns.
        max_discrete_levels: Largest number of distinct numeric values treated as categories.
        min_count: Cells with fewer records are set to missing in the rate table.

    Returns:
        Tuple (bad-rate table, record-count table), both levels(variable) x levels(stratum),
        in numeric order where possible; rows and columns without any cell
        reaching ``min_count`` are removed.
    """
    rows = to_strata(metadata[variable], n_bins, top_n, max_discrete_levels).rename(variable)
    columns = to_strata(metadata[stratum], n_bins, top_n, max_discrete_levels).rename(stratum)
    labels = metadata[label_column].astype(int)
    counts = sort_levels(pd.crosstab(rows, columns))
    rates = pd.crosstab(rows, columns, values=labels, aggfunc="mean").reindex_like(counts)
    rates = rates.where(counts >= min_count)
    kept_rows = rates.notna().any(axis=1)
    kept_columns = rates.notna().any(axis=0)
    return rates.loc[kept_rows, kept_columns], counts.loc[kept_rows, kept_columns]


def linear_bin_edges(start: float, stop: float, step: float) -> np.ndarray:
    """Equally spaced bin edges from ``start`` to ``stop`` (inclusive).

    Args:
        start: First edge.
        stop: Last edge.
        step: Bin width.

    Returns:
        Array of edges.
    """
    return np.arange(start, stop + step / 2.0, step)


def log_bin_edges(start: float, stop: float, n_bins: int) -> np.ndarray:
    """Logarithmically spaced bin edges.

    Args:
        start: First edge (positive).
        stop: Last edge.
        n_bins: Number of bins.

    Returns:
        Array of ``n_bins + 1`` edges.
    """
    return np.logspace(np.log10(start), np.log10(stop), n_bins + 1)


def binned_bad_rate_2d(
    metadata: pd.DataFrame,
    x_column: str,
    y_column: str,
    label_column: str,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    min_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bad rate in the cells of a two-dimensional grid of two numeric variables.

    Args:
        metadata: Combined metadata.
        x_column: Variable binned along the first axis.
        y_column: Variable binned along the second axis.
        label_column: Label column (1 = bad).
        x_edges: Bin edges of ``x_column``.
        y_edges: Bin edges of ``y_column``.
        min_count: Cells with fewer records are set to NaN in the rate array.

    Returns:
        Tuple (rates, counts), both of shape (len(x_edges) - 1, len(y_edges) - 1).
    """
    x = pd.to_numeric(metadata[x_column], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(metadata[y_column], errors="coerce").to_numpy(dtype=float)
    is_bad = metadata[label_column].to_numpy() == 1
    valid = np.isfinite(x) & np.isfinite(y)
    counts, _, _ = np.histogram2d(x[valid], y[valid], bins=[x_edges, y_edges])
    bad_counts, _, _ = np.histogram2d(x[valid & is_bad], y[valid & is_bad], bins=[x_edges, y_edges])
    rates = np.full(counts.shape, np.nan)
    reliable = counts >= min_count
    rates[reliable] = bad_counts[reliable] / counts[reliable]
    return rates, counts


def group_bad_rate(
    metadata: pd.DataFrame, group_column: str, label_column: str, value_columns: Sequence[str]
) -> pd.DataFrame:
    """Number of records and bad rate per group, with the median of other columns.

    Used to summarize stations (with their coordinates) or events (with
    epicentre coordinates and magnitude).

    Args:
        metadata: Combined metadata.
        group_column: Grouping column (e.g. station or event identifier).
        label_column: Label column (1 = bad).
        value_columns: Numeric columns summarized by their median within each group.

    Returns:
        Table indexed by group with ``n_records``, ``bad_rate`` and the value columns.
    """
    grouped = metadata.groupby(group_column, observed=True)
    table = pd.DataFrame({"n_records": grouped.size(), "bad_rate": grouped[label_column].mean()})
    medians = grouped[list(value_columns)].median(numeric_only=True)
    return table.join(medians)


def records_per_year_by_label(metadata: pd.DataFrame, year_column: str, label_column: str) -> pd.DataFrame:
    """Number of records per year and label, including years without records.

    Args:
        metadata: Combined metadata.
        year_column: Integer year column.
        label_column: Label column.

    Returns:
        Table (years x labels) of record counts.
    """
    years = pd.to_numeric(metadata[year_column], errors="coerce")
    table = pd.crosstab(years, metadata[label_column])
    all_years = np.arange(int(years.min()), int(years.max()) + 1)
    return table.reindex(all_years, fill_value=0)


def geographic_extent(
    longitudes: pd.Series, latitudes: pd.Series, margin_deg: float
) -> tuple[float, float, float, float]:
    """Bounding box of a set of coordinates with a margin.

    Args:
        longitudes: Longitudes in degrees.
        latitudes: Latitudes in degrees.
        margin_deg: Margin added on every side.

    Returns:
        Tuple (lon_min, lon_max, lat_min, lat_max).
    """
    return (
        float(longitudes.min()) - margin_deg,
        float(longitudes.max()) + margin_deg,
        float(latitudes.min()) - margin_deg,
        float(latitudes.max()) + margin_deg,
    )


def count_outside_extent(
    longitudes: pd.Series, latitudes: pd.Series, extent: tuple[float, float, float, float]
) -> int:
    """Number of points outside a geographic extent.

    Args:
        longitudes: Longitudes in degrees.
        latitudes: Latitudes in degrees.
        extent: (lon_min, lon_max, lat_min, lat_max).

    Returns:
        Number of points with coordinates outside the extent.
    """
    lon_min, lon_max, lat_min, lat_max = extent
    inside = longitudes.between(lon_min, lon_max) & latitudes.between(lat_min, lat_max)
    return int((~inside).sum())
