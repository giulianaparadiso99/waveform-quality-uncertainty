"""Random selection of records for development subsets, inspections and class balancing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wqu.config import NO_CLASS_BALANCING, UNDERSAMPLE_CLASS_BALANCING


def sample_records_per_label(
    metadata: pd.DataFrame,
    label_column: str,
    group_column: str,
    records_per_label: int | None,
    max_records_per_group: int | None,
    seed: int,
) -> pd.DataFrame:
    """Draw records at random for each label, limiting records from the same group.

    Records are shuffled, at most ``max_records_per_group`` are kept per group
    (e.g. per event, to favour variety), and the first ``records_per_label``
    records of each label are returned.

    Args:
        metadata: Combined metadata.
        label_column: Label column.
        group_column: Grouping column limited by ``max_records_per_group``.
        records_per_label: Records drawn per label; None keeps every eligible record.
        max_records_per_group: Maximum records per group and label; None means no limit.
        seed: Random seed.

    Returns:
        Selected rows (original index preserved), ordered by label then draw order.

    Raises:
        ValueError: If a label has fewer eligible records than requested.
    """
    rng = np.random.default_rng(seed)
    shuffled = metadata.iloc[rng.permutation(len(metadata))]
    if max_records_per_group is not None:
        rank_in_group = shuffled.groupby([label_column, group_column], observed=True).cumcount()
        shuffled = shuffled[rank_in_group < max_records_per_group]

    selections = []
    for label, records in shuffled.groupby(label_column, sort=True):
        if records_per_label is not None and len(records) < records_per_label:
            raise ValueError(
                f"Label {label}: {len(records)} eligible records, {records_per_label} requested."
            )
        selections.append(records if records_per_label is None else records.iloc[:records_per_label])
    return pd.concat(selections)


def undersample_per_group(records: pd.DataFrame, label_column: str, group_column: str, seed: int) -> pd.DataFrame:
    """Balance the labels inside each group by undersampling.

    In every group (e.g. the training and the validation set), all the records of the rarest label are
    kept and as many records of each other label are drawn at random.

    Args:
        records: Records with a label and a group column.
        label_column: Label column.
        group_column: Column of the groups balanced separately.
        seed: Random seed (the same for every group).

    Returns:
        Balanced records, in their original order (original index preserved).

    Raises:
        ValueError: If a group holds a single label.
    """
    balanced = []
    for group, group_records in records.groupby(group_column, sort=True):
        label_counts = group_records[label_column].value_counts()
        if len(label_counts) < 2:
            raise ValueError(f"Group {group!r} holds a single label: it cannot be balanced.")
        balanced.append(sample_records_per_label(group_records, label_column, label_column,
                                                 int(label_counts.min()), None, seed))
    return pd.concat(balanced).sort_index()


def balance_classes(records: pd.DataFrame, balancing: str, label_column: str, group_column: str,
                    seed: int) -> pd.DataFrame:
    """Records after the configured class balancing (``training.class_balancing``).

    Args:
        records: Records with a label and a group column.
        balancing: ``none`` (records returned unchanged) or ``undersample`` (``undersample_per_group``).
        label_column: Label column.
        group_column: Column of the groups balanced separately (the split).
        seed: Random seed.

    Returns:
        Records used by the experiment.

    Raises:
        ValueError: If the balancing is unknown.
    """
    if balancing == NO_CLASS_BALANCING:
        return records
    if balancing == UNDERSAMPLE_CLASS_BALANCING:
        return undersample_per_group(records, label_column, group_column, seed)
    raise ValueError(f"Unknown class balancing {balancing!r}.")
