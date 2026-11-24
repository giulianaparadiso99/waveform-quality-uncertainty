"""Random selection of records for development subsets and inspections."""

from __future__ import annotations

import numpy as np
import pandas as pd


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
