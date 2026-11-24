"""Reading single traces from the remote SeisBench waveform files.

SeisBench stores the traces of a dataset in one HDF5 file, packed in blocks:
the trace name ``bucket3$17`` refers to element 17 of the array
``data/bucket3``. The server accepts HTTP range requests, so a single trace
(about 1 MB for ESM25 CV records) can be read without downloading the whole
file. This is used to build a small local development subset before the full
download is available; it is far too slow for training.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wqu.config import ProjectConfig, SubsetConfig
from wqu.data.sampling import sample_records_per_label

DATA_GROUP = "data"
TRACE_NAME_COLUMN = "trace_name"
WAVEFORM_DATASET = "waveforms"
DEV_INDEX_COLUMN = "dev_index"
REMOTE_BLOCK_SIZE_BYTES = 2**21


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
def open_remote_hdf5(url: str) -> Iterator[h5py.File]:
    """Open a remote (or local) HDF5 file for random-access reading.

    Args:
        url: HTTP(S) URL or local path.

    Yields:
        Open h5py File; closed when the context exits.
    """
    with fsspec.open(url, mode="rb", block_size=REMOTE_BLOCK_SIZE_BYTES, cache_type="readahead") as handle:
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
