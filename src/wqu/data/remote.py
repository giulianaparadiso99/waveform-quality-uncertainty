"""Reading single traces from the remote SeisBench waveform files.

SeisBench stores the traces of a dataset in one HDF5 file, packed in blocks:
the trace name ``bucket3$17`` refers to element 17 of the array
``data/bucket3``. The server accepts HTTP range requests, so a single trace
(about 1 MB for ESM25 CV records) can be read without downloading the whole
file. This is used to build a small local development subset before the full
download is available, and to compute features from datasets too large to be
downloaded (``iter_remote_trace_batches``: the ESM25GoodMP acceleration).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

import aiohttp
import fsspec
import h5py
import numpy as np
import pandas as pd
from fsspec.exceptions import FSTimeoutError
from tqdm.auto import tqdm

from wqu.config import ProjectConfig, SubsetConfig
from wqu.data.sampling import sample_records_per_label

DATA_GROUP = "data"
DATA_FORMAT_GROUP = "data_format"
COMPONENT_ORDER_KEY = "component_order"
TRACE_NAME_COLUMN = "trace_name"
WAVEFORM_DATASET = "waveforms"
DEV_INDEX_COLUMN = "dev_index"
REMOTE_BLOCK_SIZE_BYTES = 2**21
# Network errors after which a batch is read again from a newly opened file.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (FSTimeoutError, asyncio.TimeoutError, aiohttp.ClientError, OSError)
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_WAIT_S = 10.0

logger = logging.getLogger(__name__)


def parse_trace_name(trace_name: str) -> tuple[str, tuple[int | slice, ...]]:
    """Split a SeisBench trace name into block name and location inside the block.

    Supported forms: ``"bucket3$17"`` and ``"bucket3$17,:3,:84000"``.

    Args:
        trace_name: Trace name from the SeisBench metadata.

    Returns:
        Tuple (block name, index tuple usable on the HDF5 dataset).

    Raises:
        ValueError: If the name does not contain a block location.
    """
    if "$" not in trace_name:
        raise ValueError(f"Trace name '{trace_name}' does not reference a block ('$' missing).")
    block_name, location = trace_name.split("$", 1)
    index: list[int | slice] = []
    for part in location.split(","):
        if ":" in part:
            start, stop = part.split(":", 1)
            index.append(slice(int(start) if start else None, int(stop) if stop else None))
        else:
            index.append(int(part))
    return block_name, tuple(index)


def remote_waveform_url(template: str, subset: SubsetConfig) -> str:
    """Return the URL of the waveform file of a subset.

    Args:
        template: URL template containing ``{repository_name}``.
        subset: Subset configuration.

    Returns:
        URL string.
    """
    return template.format(repository_name=subset.repository_name)


@contextmanager
def open_remote_hdf5(url: str, block_size_bytes: int = REMOTE_BLOCK_SIZE_BYTES) -> Iterator[h5py.File]:
    """Open a remote (or local) HDF5 file for random-access reading.

    Args:
        url: HTTP(S) URL or local path.
        block_size_bytes: Size of each range request; with the read-ahead cache, larger blocks make
            sequential reads (traces in storage order) faster.

    Yields:
        Open h5py File; closed when the context exits.
    """
    with fsspec.open(url, mode="rb", block_size=block_size_bytes, cache_type="readahead") as handle:
        with h5py.File(handle, mode="r") as hdf5_file:
            yield hdf5_file


def read_trace(hdf5_file: h5py.File, trace_name: str, expected_shape: tuple[int, int]) -> np.ndarray:
    """Read one trace from an open SeisBench waveform file.

    Args:
        hdf5_file: Open SeisBench waveform file.
        trace_name: Trace name from the metadata.
        expected_shape: Expected (components, samples) shape.

    Returns:
        Float32 array of shape ``expected_shape``.

    Raises:
        ValueError: If the trace has an unexpected shape.
    """
    block_name, index = parse_trace_name(trace_name)
    trace = np.asarray(hdf5_file[DATA_GROUP][block_name][index], dtype=np.float32)
    if trace.shape != expected_shape:
        raise ValueError(f"Trace '{trace_name}' has shape {trace.shape}, expected {expected_shape}.")
    return trace


def file_component_order(hdf5_file: h5py.File) -> tuple[str, ...]:
    """Component order stored in the ``data_format`` group of a SeisBench waveform file.

    Args:
        hdf5_file: Open SeisBench waveform file.

    Returns:
        Component names in storage order (e.g. ``("acc_mp_u", ..., "dis_mp_w")``): an array of names,
        a comma-separated string or a string of one-letter components (``"ZNE"``).

    Raises:
        KeyError: If the file does not store a component order.
    """
    if DATA_FORMAT_GROUP not in hdf5_file or COMPONENT_ORDER_KEY not in hdf5_file[DATA_FORMAT_GROUP]:
        raise KeyError(f"No {DATA_FORMAT_GROUP}/{COMPONENT_ORDER_KEY} in the waveform file.")
    value = hdf5_file[DATA_FORMAT_GROUP][COMPONENT_ORDER_KEY][()]
    items = value if isinstance(value, np.ndarray) else [value]
    names = tuple(item.decode() if isinstance(item, bytes) else str(item) for item in np.ravel(items))
    if len(names) > 1:
        return names
    single = names[0]
    return tuple(single.split(",")) if "," in single else tuple(single)


def component_positions(stored_order: Sequence[str], components: Sequence[str]) -> list[int]:
    """Positions of the requested components in the stored order.

    Args:
        stored_order: Component names in storage order.
        components: Requested components, in the order wanted.

    Returns:
        One position per requested component.

    Raises:
        KeyError: If a requested component is not stored.
    """
    missing = [component for component in components if component not in stored_order]
    if missing:
        raise KeyError(f"Components {missing} not in the stored order {tuple(stored_order)}.")
    return [list(stored_order).index(component) for component in components]


def read_trace_components(hdf5_file: h5py.File, trace_name: str, positions: Sequence[int]) -> np.ndarray:
    """Read some components of one trace from an open SeisBench waveform file.

    The trace name may give the components and samples of the trace (``bucket3$17,:9,:84000``);
    the requested positions are taken among those components.

    Args:
        hdf5_file: Open SeisBench waveform file (dimension order components x samples).
        trace_name: Trace name from the metadata.
        positions: Positions of the components to keep, in the stored order.

    Returns:
        Float32 array of shape (len(positions), samples).
    """
    block_name, index = parse_trace_name(trace_name)
    row = index[0]
    samples = index[2] if len(index) > 2 else slice(None)
    channels = sorted(set(positions))
    block = hdf5_file[DATA_GROUP][block_name]
    if channels == list(range(channels[0], channels[-1] + 1)):
        stored = np.asarray(block[row, channels[0]:channels[-1] + 1, samples], dtype=np.float32)
    else:
        stored = np.asarray(block[row, channels, samples], dtype=np.float32)
    return stored[[channels.index(position) for position in positions]]


def iter_remote_trace_batches(
    records: pd.DataFrame,
    url: str,
    components: Sequence[str],
    batch_size: int,
    block_size_bytes: int = REMOTE_BLOCK_SIZE_BYTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    """Yield batches of records and some components of their traces, read from a remote waveform file.

    Records are read in storage order (block, then position in the block), so that the read-ahead
    cache serves consecutive traces with few requests. When a batch fails with a network error
    (``TRANSIENT_ERRORS``), the file is opened again and the batch is read again, after a wait that
    doubles at every attempt.

    Args:
        records: Records with a ``trace_name`` column referring to the remote file.
        url: URL of the SeisBench waveform file.
        components: Components to read (e.g. ``acc_mp_u``, ``acc_mp_v``, ``acc_mp_w``).
        batch_size: Records per batch.
        block_size_bytes: Size of each range request.
        max_retries: New attempts for a failed batch before the error is raised.
        retry_wait_s: Wait before the first new attempt.

    Yields:
        Tuples (records of the batch, array of shape (batch, components, samples)).

    Raises:
        Exception: The network error of the last attempt, when all the attempts fail.
    """
    locations = records[TRACE_NAME_COLUMN].map(parse_trace_name)
    order = sorted(range(len(records)), key=lambda position: (locations.iloc[position][0], locations.iloc[position][1][0]))
    ordered = records.iloc[order]
    stack = ExitStack()
    try:
        remote = stack.enter_context(open_remote_hdf5(url, block_size_bytes))
        positions = component_positions(file_component_order(remote), components)
        for start in range(0, len(ordered), batch_size):
            batch = ordered.iloc[start:start + batch_size]
            for attempt in range(max_retries + 1):
                try:
                    traces = [read_trace_components(remote, name, positions) for name in batch[TRACE_NAME_COLUMN]]
                    break
                except TRANSIENT_ERRORS as error:
                    if attempt == max_retries:
                        raise
                    wait_s = retry_wait_s * 2**attempt
                    logger.warning("Batch at record %d failed (%s); new attempt in %.0f s.", start, type(error).__name__, wait_s)
                    stack.close()
                    time.sleep(wait_s)
                    stack = ExitStack()
                    remote = stack.enter_context(open_remote_hdf5(url, block_size_bytes))
            yield batch, np.stack(traces)
    finally:
        stack.close()


def select_dev_subset(metadata: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Choose the records of the development subset.

    Args:
        metadata: Combined metadata.
        config: Project configuration.

    Returns:
        Selected records with columns subset, subset index, label, trace name and
        ``dev_index`` (row of the trace in the local subset file).
    """
    columns = config.columns
    selected = sample_records_per_label(
        metadata,
        label_column=columns.label,
        group_column=columns.event_id,
        records_per_label=config.waveforms.dev_subset_records_per_class,
        max_records_per_group=config.waveforms.dev_subset_max_records_per_event,
        seed=config.waveforms.random_seed,
    )
    records = selected[[columns.subset, columns.subset_index, columns.label, TRACE_NAME_COLUMN]].copy()
    records[columns.subset] = records[columns.subset].astype(str)
    records[DEV_INDEX_COLUMN] = np.arange(len(records))
    return records.reset_index(drop=True)


def build_dev_subset(
    metadata: pd.DataFrame, config: ProjectConfig, expected_shape: tuple[int, int]
) -> pd.DataFrame:
    """Read the development subset from the server and save it locally.

    The traces are written to a temporary file renamed at the end, so an
    interrupted run leaves no incomplete subset behind (it restarts from zero).

    Args:
        metadata: Combined metadata.
        config: Project configuration.
        expected_shape: Expected (components, samples) shape of a trace.

    Returns:
        Table of the stored records (see ``select_dev_subset``).
    """
    waveforms = config.waveforms
    records = select_dev_subset(metadata, config)
    waveforms.dev_subset_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = waveforms.dev_subset_file.with_suffix(".partial.h5")
    subsets_by_name = {subset.name: subset for subset in config.data.subsets}

    with h5py.File(temporary_file, mode="w") as output:
        stored = output.create_dataset(
            WAVEFORM_DATASET, shape=(len(records), *expected_shape), dtype="float32", chunks=(1, *expected_shape)
        )
        for subset_name, subset_records in records.groupby(config.columns.subset, sort=False):
            url = remote_waveform_url(waveforms.remote_url_template, subsets_by_name[subset_name])
            with open_remote_hdf5(url) as remote:
                for record in tqdm(subset_records.itertuples(index=False), total=len(subset_records), desc=subset_name):
                    trace_name = getattr(record, TRACE_NAME_COLUMN)
                    stored[getattr(record, DEV_INDEX_COLUMN)] = read_trace(remote, trace_name, expected_shape)

    temporary_file.replace(waveforms.dev_subset_file)
    records.to_csv(waveforms.dev_subset_records_file, index=False)
    return records


def load_dev_subset_records(path: Path) -> pd.DataFrame:
    """Read the table of records stored in the development subset.

    Args:
        path: CSV written by ``build_dev_subset``.

    Returns:
        Records table with ``trace_name`` read as text.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Development subset records not found: {path}")
    return pd.read_csv(path, dtype={TRACE_NAME_COLUMN: "string"})


def ensure_dev_subset(
    metadata: pd.DataFrame, config: ProjectConfig, expected_shape: tuple[int, int]
) -> pd.DataFrame:
    """Return the development subset records, building the subset if it does not exist.

    Args:
        metadata: Combined metadata.
        config: Project configuration.
        expected_shape: Expected (components, samples) shape of a trace.

    Returns:
        Records table of the subset.
    """
    waveforms = config.waveforms
    if waveforms.dev_subset_file.is_file() and waveforms.dev_subset_records_file.is_file():
        return load_dev_subset_records(waveforms.dev_subset_records_file)
    return build_dev_subset(metadata, config, expected_shape)
