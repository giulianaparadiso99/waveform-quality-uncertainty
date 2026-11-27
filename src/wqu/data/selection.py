"""Selection of the records of an experiment by network, channel, orientation and magnitude.

Shalabi et al. (2026) train on the three-component accelerograms (HNE, HNN, HNZ channels) of the
ITACA, HEAD and TR-NSMN archives, with magnitude of at least 4. ``training.record_selection``
reproduces such a selection on ESM25: a record is kept when it meets every configured criterion,
and an empty criterion keeps all records. The criteria are applied in a fixed order (network,
channel, orientation, magnitude), so that the records left after each of them can be counted.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from wqu.config import RecordSelectionConfig

NETWORK_COLUMN = "station_network_code"
CHANNEL_COLUMN = "station_channel_code"
# Orientation code of the u, v and w components, in this order.
ORIENTATION_COLUMNS: tuple[str, ...] = ("trace_u_channel_code", "trace_v_channel_code", "trace_w_channel_code")
MAGNITUDE_COLUMN = "preferred_magnitude"
ALL_RECORDS_STEP = "all records"


def selection_columns(selection: RecordSelectionConfig) -> list[str]:
    """Columns of the record table needed by the configured criteria.

    Args:
        selection: Record selection.

    Returns:
        Column names, in the order of the criteria.
    """
    columns = []
    if selection.networks:
        columns.append(NETWORK_COLUMN)
    if selection.channel_code is not None:
        columns.append(CHANNEL_COLUMN)
    if selection.orientations:
        columns.extend(ORIENTATION_COLUMNS)
    if selection.min_magnitude is not None:
        columns.append(MAGNITUDE_COLUMN)
    return columns


def check_selection_columns(records: pd.DataFrame, selection: RecordSelectionConfig) -> None:
    """Check that the record table has the columns needed by the selection.

    Args:
        records: Record table.
        selection: Record selection.

    Raises:
        KeyError: If a needed column is missing.
    """
    missing = [column for column in selection_columns(selection) if column not in records.columns]
    if missing:
        raise KeyError(f"The record table lacks the columns needed by training.record_selection: {missing}")


def network_mask(records: pd.DataFrame, networks: tuple[str, ...]) -> pd.Series:
    """Records of the given networks.

    Args:
        records: Record table with the network column.
        networks: Network codes to keep.

    Returns:
        Boolean mask aligned with the table.
    """
    return records[NETWORK_COLUMN].isin(networks)


def channel_mask(records: pd.DataFrame, channel_code: str) -> pd.Series:
    """Records whose channels have the given band and instrument code.

    Args:
        records: Record table with the channel column.
        channel_code: Code to keep, e.g. ``HN``.

    Returns:
        Boolean mask aligned with the table.
    """
    return records[CHANNEL_COLUMN] == channel_code


def orientation_mask(records: pd.DataFrame, orientations: tuple[str, ...]) -> pd.Series:
    """Records whose u, v and w components have the given orientation codes.

    Args:
        records: Record table with the orientation columns.
        orientations: Codes of the u, v and w components, e.g. ``(E, N, Z)``.

    Returns:
        Boolean mask aligned with the table.

    Raises:
        ValueError: If the number of codes differs from the number of components.
    """
    if len(orientations) != len(ORIENTATION_COLUMNS):
        raise ValueError(f"Expected {len(ORIENTATION_COLUMNS)} orientation codes, got {len(orientations)}.")
    mask = pd.Series(True, index=records.index)
    for column, orientation in zip(ORIENTATION_COLUMNS, orientations):
        mask &= records[column] == orientation
    return mask


def magnitude_mask(records: pd.DataFrame, min_magnitude: float) -> pd.Series:
    """Records with preferred magnitude of at least ``min_magnitude`` (records without magnitude excluded).

    Args:
        records: Record table with the magnitude column.
        min_magnitude: Smallest magnitude to keep.

    Returns:
        Boolean mask aligned with the table.
    """
    return records[MAGNITUDE_COLUMN].astype(float) >= min_magnitude


def selection_steps(records: pd.DataFrame, selection: RecordSelectionConfig) -> dict[str, pd.Series]:
    """Cumulative masks of the configured criteria, in the order network, channel, orientation, magnitude.

    Args:
        records: Record table.
        selection: Record selection.

    Returns:
        Ordered mapping from step name to the mask of the records that meet that criterion and all the
        previous ones; the first step keeps all records, the last one is the selection.
    """
    check_selection_columns(records, selection)
    current = pd.Series(True, index=records.index)
    steps = {ALL_RECORDS_STEP: current}
    criteria = []
    if selection.networks:
        criteria.append((f"networks {', '.join(selection.networks)}", network_mask(records, selection.networks)))
    if selection.channel_code is not None:
        criteria.append((f"channel {selection.channel_code}", channel_mask(records, selection.channel_code)))
    if selection.orientations:
        criteria.append((f"orientations {', '.join(selection.orientations)}",
                         orientation_mask(records, selection.orientations)))
    if selection.min_magnitude is not None:
        criteria.append((f"magnitude >= {selection.min_magnitude:g}", magnitude_mask(records, selection.min_magnitude)))
    for name, mask in criteria:
        current = current & mask
        steps[f"+ {name}"] = current
    return steps


def selection_mask(records: pd.DataFrame, selection: RecordSelectionConfig) -> pd.Series:
    """Records kept by the selection.

    Args:
        records: Record table.
        selection: Record selection.

    Returns:
        Boolean mask aligned with the table (all true when the selection is empty).
    """
    return list(selection_steps(records, selection).values())[-1]


def select_records(records: pd.DataFrame, selection: RecordSelectionConfig) -> pd.DataFrame:
    """Records kept by the selection, with their original index.

    Args:
        records: Record table.
        selection: Record selection.

    Returns:
        Subset of the table.
    """
    return records[selection_mask(records, selection)]


def class_counts(labels: pd.Series, groups: pd.Series, class_names: Mapping[int, str]) -> pd.DataFrame:
    """Records per group and class, with the total and the fraction of bad records.

    Args:
        labels: Class label of every record.
        groups: Group of every record, aligned with the labels (missing values counted as ``missing``).
        class_names: Mapping from label to class name; the bad class must be named ``bad``.

    Returns:
        Table indexed by group, sorted by decreasing total.
    """
    values = groups.astype("string").fillna("missing")
    counts = pd.crosstab(values, labels.map(class_names))
    counts = counts.reindex(columns=list(class_names.values()), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["bad_fraction"] = counts["bad"] / counts["total"]
    return counts.sort_values("total", ascending=False)


def step_counts(labels: pd.Series, steps: Mapping[str, pd.Series], class_names: Mapping[int, str]) -> pd.DataFrame:
    """Records per class left after each step of the selection.

    Args:
        labels: Class label of every record.
        steps: Output of ``selection_steps``, aligned with the labels.
        class_names: Mapping from label to class name; the bad class must be named ``bad``.

    Returns:
        Table with one row per step: records per class, total, fraction of bad records and share of
        all records kept.
    """
    rows = {}
    for name, mask in steps.items():
        counts = labels[mask].map(class_names).value_counts()
        rows[name] = {class_name: int(counts.get(class_name, 0)) for class_name in class_names.values()}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table["total"] = table.sum(axis=1)
    table["bad_fraction"] = table["bad"] / table["total"]
    table["share_kept"] = table["total"] / len(labels)
    return table
