"""Access to ESM25 waveforms from the development subset or the full SeisBench datasets.

The inspection and preprocessing code only sees batches of
(records, waveforms), so switching ``waveforms.source`` in the configuration
from ``dev_subset`` to ``seisbench`` runs the same analysis on the full data.
"""

from __future__ import annotations

from collections.abc import Iterator

import h5py
import numpy as np
import pandas as pd

from wqu.config import ProjectConfig
from wqu.data.loading import CV_COMPONENTS, load_seisbench_datasets
from wqu.data.remote import DEV_INDEX_COLUMN, WAVEFORM_DATASET, ensure_dev_subset
from wqu.data.sampling import sample_records_per_label

STORED_SAMPLING_RATE_COLUMN = "trace_sampling_rate_hz"


def stored_sampling_rate_hz(metadata: pd.DataFrame) -> float:
    """Return the common sampling rate of the stored ESM25 traces.

    Args:
        metadata: Combined metadata.

    Returns:
        Sampling rate in hertz.

    Raises:
        ValueError: If the traces do not share a single sampling rate.
    """
    rates = pd.to_numeric(metadata[STORED_SAMPLING_RATE_COLUMN], errors="coerce").dropna().unique()
    if len(rates) != 1:
        raise ValueError(f"Expected one stored sampling rate, found {sorted(rates)}.")
    return float(rates[0])


def expected_trace_shape(metadata: pd.DataFrame, waveform_length_s: float) -> tuple[int, int]:
    """Shape (components, samples) of a stored ESM25 CV trace.

    Args:
        metadata: Combined metadata.
        waveform_length_s: Fixed trace length in seconds.

    Returns:
        Tuple (number of components, number of samples).
    """
    return len(CV_COMPONENTS), int(round(waveform_length_s * stored_sampling_rate_hz(metadata)))


def select_inspection_records(metadata: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Records analysed by the waveform inspection, according to ``waveforms.source``.

    - ``dev_subset``: the records of the local development subset (built on first use);
    - ``seisbench``: all records, or ``inspection_records_per_class`` random records per class.

    Args:
        metadata: Combined metadata.
        config: Project configuration.

    Returns:
        Rows of the combined metadata (original index preserved); in
        ``dev_subset`` mode with the extra ``dev_index`` column.
    """
    columns = config.columns
    waveforms = config.waveforms
    if waveforms.source == "dev_subset":
        shape = expected_trace_shape(metadata, config.data.waveform_length_s)
        stored = ensure_dev_subset(metadata, config, shape)
        keys = metadata[[columns.subset, columns.subset_index]].astype({columns.subset: str})
        positions = pd.MultiIndex.from_frame(keys).get_indexer(
            pd.MultiIndex.from_frame(stored[[columns.subset, columns.subset_index]].astype({columns.subset: str}))
        )
        if (positions < 0).any():
            raise KeyError("Some development subset records are missing from the metadata.")
        selected = metadata.iloc[positions].copy()
        selected[DEV_INDEX_COLUMN] = stored[DEV_INDEX_COLUMN].to_numpy()
        return selected
    return sample_records_per_label(
        metadata,
        label_column=columns.label,
        group_column=columns.event_id,
        records_per_label=waveforms.inspection_records_per_class,
        max_records_per_group=None,
        seed=waveforms.random_seed,
    )


def _iter_dev_subset_batches(records: pd.DataFrame, config: ProjectConfig) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield batches of records and traces from the local development subset.

    Args:
        records: Records with a ``dev_index`` column.
        config: Project configuration.

    Yields:
        Tuples (records of the batch, array of shape (batch, components, samples)).
    """
    ordered = records.sort_values(DEV_INDEX_COLUMN)
    batch_size = config.waveforms.batch_size
    with h5py.File(config.waveforms.dev_subset_file, mode="r") as stored:
        traces = stored[WAVEFORM_DATASET]
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start : start + batch_size]
            yield batch, traces[batch[DEV_INDEX_COLUMN].to_numpy()].astype(np.float32)


def _iter_seisbench_batches(records: pd.DataFrame, config: ProjectConfig) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield batches of records and traces from the SeisBench datasets in the cache.

    Args:
        records: Rows of the combined metadata.
        config: Project configuration.

    Yields:
        Tuples (records of the batch, array of shape (batch, components, samples)).
    """
    columns = config.columns
    datasets = load_seisbench_datasets(config)
    batch_size = config.waveforms.batch_size
    for subset_name, subset_records in records.groupby(columns.subset, observed=True, sort=False):
        ordered = subset_records.sort_values(columns.subset_index)
        dataset = datasets[str(subset_name)]
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start : start + batch_size]
            traces = dataset.get_waveforms(batch[columns.subset_index].astype(int).tolist())
            yield batch, np.asarray(traces, dtype=np.float32)


def iter_waveform_batches(records: pd.DataFrame, config: ProjectConfig) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield batches of records and their traces from the configured waveform source.

    Args:
        records: Output of ``select_inspection_records`` (or a subset of it).
        config: Project configuration.

    Yields:
        Tuples (records of the batch, array of shape (batch, components, samples)).
    """
    if config.waveforms.source == "dev_subset":
        yield from _iter_dev_subset_batches(records, config)
    else:
        yield from _iter_seisbench_batches(records, config)


def load_waveforms(records: pd.DataFrame, config: ProjectConfig) -> np.ndarray:
    """Load the traces of a few records, in the order of ``records``.

    Args:
        records: Records to load (small number, e.g. examples for plots).
        config: Project configuration.

    Returns:
        Array of shape (records, components, samples).
    """
    by_index = {}
    for batch, traces in iter_waveform_batches(records, config):
        by_index.update(zip(batch.index, traces))
    return np.stack([by_index[index] for index in records.index])
