"""Training/validation split of ESM25, grouped by event and stratified by label.

All records of an event fall in the same set, so the validation set contains only
events never seen in training, as the test set (later ESM records) will. The
stations, instead, are largely shared between the sets, as they will be with the
test set; ``station_overlap`` measures how much.

The split is stored as a CSV file keyed by subset and position in the subset, so
that it can be joined with any table derived from the metadata.

Cross-validation folds (``assign_folds``) are built with the same method on the records of
the split table, in its order: with as many folds as the split, fold 1 is its validation set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from wqu.config import EVENT_SPLIT_GROUPING, ColumnsConfig, SplitConfig

SPLIT_COLUMN = "split"
FOLD_COLUMN = "fold"
TRAIN = "train"
VALIDATION = "validation"


def fold_numbers(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold of every record (1 to ``n_folds``), stratified by label and grouped by event.

    Args:
        labels: Record labels.
        groups: Group of each record (event); a group lies in one fold.
        n_folds: Number of folds.
        seed: Random seed of the fold assignment.

    Returns:
        Integer array of fold numbers.
    """
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.zeros(len(labels), dtype=int)
    for fold, (_, validation_positions) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups), start=1):
        folds[validation_positions] = fold
    return folds


def validation_mask(labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Mark the records of the validation set: the first of ``n_folds`` folds (see ``fold_numbers``).

    Args:
        labels: Record labels.
        groups: Group of each record (event).
        n_folds: Number of folds.
        seed: Random seed of the fold assignment.

    Returns:
        Boolean array, true for validation records.
    """
    return fold_numbers(labels, groups, n_folds, seed) == 1


def assign_folds(split_table: pd.DataFrame, columns: ColumnsConfig, n_folds: int, seed: int) -> pd.DataFrame:
    """Cross-validation folds of the records of a split table, in the order of the table.

    Args:
        split_table: Output of ``assign_split`` or ``load_split`` (all ESM25 records).
        columns: Column configuration.
        n_folds: Number of folds.
        seed: Random seed (``split.random_seed`` reproduces the split when ``n_folds`` equals its folds).

    Returns:
        Table with subset, subset index and ``fold``.
    """
    table = split_table[[columns.subset, columns.subset_index]].copy()
    table[FOLD_COLUMN] = fold_numbers(
        split_table[columns.label].to_numpy(), split_table[columns.event_id].to_numpy(), n_folds, seed
    )
    return table


def split_groups(metadata: pd.DataFrame, columns: ColumnsConfig, grouping: str) -> np.ndarray:
    """Group of every record for the split: its event, or the record itself (no grouping).

    Args:
        metadata: Records with the event column.
        columns: Column configuration.
        grouping: ``event`` or ``record`` (``wqu.config.VALID_SPLIT_GROUPINGS``).

    Returns:
        Array of group identifiers aligned with ``metadata``.

    Raises:
        ValueError: If an event identifier is missing with the event grouping.
    """
    if grouping != EVENT_SPLIT_GROUPING:
        return np.arange(len(metadata))
    if metadata[columns.event_id].isna().any():
        raise ValueError(f"Missing values in '{columns.event_id}': every record needs an event for the grouping.")
    return metadata[columns.event_id].to_numpy()


def assign_split(metadata: pd.DataFrame, columns: ColumnsConfig, split: SplitConfig) -> pd.DataFrame:
    """Assign every record to the training or validation set.

    The split is stratified by label and grouped as ``split.grouping`` says: by event (all the
    records of an event in the same set) or by record (no grouping).

    Args:
        metadata: Combined ESM25 metadata (see ``load_esm25_metadata``).
        columns: Column configuration.
        split: Split configuration.

    Returns:
        Table indexed like ``metadata`` with subset, subset index, event, station,
        label and ``split`` (``train`` or ``validation``).
    """
    mask = validation_mask(
        metadata[columns.label].to_numpy(), split_groups(metadata, columns, split.grouping), split.n_folds,
        split.random_seed,
    )
    table = metadata[[columns.subset, columns.subset_index, columns.event_id, columns.station_id, columns.label]].copy()
    table[SPLIT_COLUMN] = np.where(mask, VALIDATION, TRAIN)
    return table


def save_split(table: pd.DataFrame, path: Path) -> Path:
    """Write the split table to CSV.

    Args:
        table: Output of ``assign_split``.
        path: Destination file.

    Returns:
        The destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def load_split(path: Path, columns: ColumnsConfig) -> pd.DataFrame:
    """Read a split table written by ``save_split``.

    Args:
        path: CSV file.
        columns: Column configuration.

    Returns:
        Split table with identifiers read as text.
    """
    return pd.read_csv(path, dtype={columns.event_id: str, columns.station_id: str, columns.subset: str})


def attach_column(metadata: pd.DataFrame, table: pd.DataFrame, column: str, columns: ColumnsConfig) -> pd.Series:
    """Values of a column of a keyed table for every record of a metadata table.

    Records are matched on subset and subset index.

    Args:
        metadata: Table with the subset and subset-index columns.
        table: Table with the same keys and ``column``.
        column: Column to attach.
        columns: Column configuration.

    Returns:
        Series aligned with ``metadata``.

    Raises:
        ValueError: If some records have no value.
    """
    keys = [columns.subset, columns.subset_index]
    lookup = table[[*keys, column]].astype({columns.subset: str, columns.subset_index: int})
    records = metadata[keys].astype({columns.subset: str, columns.subset_index: int})
    matched = records.merge(lookup, on=keys, how="left")[column]
    if matched.isna().any():
        raise ValueError(f"{int(matched.isna().sum())} records have no '{column}'.")
    return pd.Series(matched.to_numpy(), index=metadata.index, name=column)


def attach_split(metadata: pd.DataFrame, split_table: pd.DataFrame, columns: ColumnsConfig) -> pd.Series:
    """Split of every record of a metadata table, matched on subset and subset index.

    Args:
        metadata: Table with the subset and subset-index columns.
        split_table: Output of ``assign_split`` or ``load_split``.
        columns: Column configuration.

    Returns:
        Series of split names aligned with ``metadata``.
    """
    return attach_column(metadata, split_table, SPLIT_COLUMN, columns)


def split_counts(table: pd.DataFrame, columns: ColumnsConfig) -> pd.DataFrame:
    """Records, share of records, bad rate, events and stations per set.

    Args:
        table: Output of ``assign_split``.
        columns: Column configuration.

    Returns:
        Table with one row per set.
    """
    grouped = table.groupby(SPLIT_COLUMN)
    summary = pd.DataFrame(
        {
            "records": grouped.size(),
            "bad_rate": grouped[columns.label].mean(),
            "events": grouped[columns.event_id].nunique(),
            "stations": grouped[columns.station_id].nunique(),
        }
    )
    summary.insert(1, "record_share", summary["records"] / summary["records"].sum())
    return summary


def shared_events(table: pd.DataFrame, columns: ColumnsConfig) -> int:
    """Number of events present in both sets (zero for a correct grouped split).

    Args:
        table: Output of ``assign_split``.
        columns: Column configuration.

    Returns:
        Number of shared events.
    """
    events = table.groupby(SPLIT_COLUMN)[columns.event_id].unique()
    return len(set(events.get(TRAIN, [])) & set(events.get(VALIDATION, [])))


def station_overlap(table: pd.DataFrame, columns: ColumnsConfig) -> dict[str, float]:
    """How much the validation stations are known from training.

    Args:
        table: Output of ``assign_split``.
        columns: Column configuration.

    Returns:
        Dictionary with the share of validation stations that also have training
        records and the share of validation records coming from such stations.
    """
    training_stations = set(table.loc[table[SPLIT_COLUMN] == TRAIN, columns.station_id])
    validation = table[table[SPLIT_COLUMN] == VALIDATION]
    known = validation[columns.station_id].isin(training_stations)
    return {
        "validation_stations_seen_in_training": float(
            validation.loc[known, columns.station_id].nunique() / validation[columns.station_id].nunique()
        ),
        "validation_records_from_seen_stations": float(known.mean()),
    }
