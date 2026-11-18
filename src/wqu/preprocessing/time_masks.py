"""Time masks of the stored features: which time bins of each window contain recorded data.

A time bin is marked as data when its centre lies between the first sample recorded by any
component and the last sample recorded by any component; outside this span the window holds
only the zero padding added to align the traces. Missing data that is not common to the three
components (a dead component, a component starting later) and gaps inside the recording are
not masked, because they are defects of the record that the network has to see.

The masks are stored in ``time_masks.h5`` next to the feature file, one row per record in the
order of ``feature_index``. A record whose mask has no data bin has no recorded samples in its
window: it cannot be classified by a network that ignores the padding.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wqu.data.inspection import edge_zero_samples
from wqu.preprocessing.pipeline import FEATURE_INDEX_COLUMN

TIME_MASK_FILE = "time_masks.h5"
TIME_MASK_DATASET = "time_mask"
WINDOW_COLUMNS: tuple[str, str] = ("window_start", "window_stop")


def time_mask_path(features_path: Path) -> Path:
    """Path of the time masks of a feature file.

    Args:
        features_path: Feature file written by ``preprocess_and_store``.

    Returns:
        ``time_masks.h5`` in the same folder.
    """
    return features_path.with_name(TIME_MASK_FILE)


def recorded_union_span(trace: np.ndarray, tolerance: float) -> tuple[int, int]:
    """Samples from the first to the last one recorded by any component.

    Args:
        trace: Stored trace, shape (components, samples).
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        Tuple (first recorded sample, sample after the last recorded one); empty (start >= stop)
        when no component has data.
    """
    leading, trailing = edge_zero_samples(trace[None], tolerance)
    n_samples = trace.shape[-1]
    return int(leading.min()), n_samples - int(trailing.min())


def time_bin_mask(recorded_start: int, recorded_stop: int, window_start: int, window_stop: int, n_times: int) -> np.ndarray:
    """Data mask of the time bins of a window.

    Args:
        recorded_start: First recorded sample (trace coordinates).
        recorded_stop: Sample after the last recorded one (trace coordinates).
        window_start: First sample of the window (trace coordinates).
        window_stop: Sample after the last one of the window (trace coordinates).
        n_times: Time bins of the window.

    Returns:
        Boolean array of shape (n_times,), True where the bin centre lies in the recorded span.
    """
    bin_length = (window_stop - window_start) / n_times
    centres = window_start + (np.arange(n_times) + 0.5) * bin_length
    return (centres >= recorded_start) & (centres < recorded_stop)


def compute_time_masks(
    stored_records: pd.DataFrame,
    batch_source: Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, np.ndarray]]],
    n_times: int,
    tolerance: float,
    output_path: Path,
) -> np.ndarray:
    """Compute the time mask of every stored record from its trace and write the mask file.

    The file is written under a temporary name and renamed at the end, so an interrupted run
    leaves no incomplete mask file (it restarts from the beginning).

    Args:
        stored_records: Record table of the feature file (``read_feature_records``), with the
            window bounds and ``feature_index``; its index matches the batches.
        batch_source: Function returning an iterable of (records, traces) for a subset of the records.
        n_times: Time bins of the feature grid.
        tolerance: Zero threshold used to find the zero padding.
        output_path: Mask file to create.

    Returns:
        Boolean array of shape (records, n_times), row ``i`` for ``feature_index == i``.

    Raises:
        RuntimeError: If some records are not returned by the batch source.
    """
    masks = np.zeros((len(stored_records), n_times), dtype=bool)
    filled = np.zeros(len(stored_records), dtype=bool)
    with tqdm(total=len(stored_records), desc="time masks") as progress:
        for batch, traces in batch_source(stored_records):
            for (index, _), trace in zip(batch.iterrows(), traces):
                record = stored_records.loc[index]
                row = int(record[FEATURE_INDEX_COLUMN])
                start, stop = recorded_union_span(trace, tolerance)
                masks[row] = time_bin_mask(start, stop, int(record[WINDOW_COLUMNS[0]]), int(record[WINDOW_COLUMNS[1]]), n_times)
                filled[row] = True
                progress.update(1)
    if not filled.all():
        raise RuntimeError(f"{int((~filled).sum())} records were not returned by the batch source.")
    write_time_masks(masks, output_path)
    return masks


def write_time_masks(masks: np.ndarray, output_path: Path) -> Path:
    """Write masks to an HDF5 file atomically (temporary name, then rename).

    Args:
        masks: Boolean array of shape (records, n_times).
        output_path: Destination file.

    Returns:
        The destination path.
    """
    temporary_path = output_path.with_suffix(".partial.h5")
    with h5py.File(temporary_path, mode="w") as output:
        output.create_dataset(TIME_MASK_DATASET, data=masks, compression="gzip")
    temporary_path.replace(output_path)
    return output_path


def read_time_masks(path: Path) -> np.ndarray:
    """Read the masks of every record.

    Args:
        path: File written by ``compute_time_masks``.

    Returns:
        Boolean array of shape (records, n_times).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist: run notebook 02c to compute the time masks.")
    with h5py.File(path, mode="r") as stored:
        return stored[TIME_MASK_DATASET][()].astype(bool)


def records_with_data(records: pd.DataFrame, masks: np.ndarray) -> pd.Series:
    """Whether each record has at least one time bin with recorded data.

    Args:
        records: Records with ``feature_index``.
        masks: Output of ``read_time_masks``.

    Returns:
        Boolean series aligned with ``records``.
    """
    rows = records[FEATURE_INDEX_COLUMN].to_numpy()
    return pd.Series(masks[rows].any(axis=1), index=records.index)


def drop_records_without_data(records: pd.DataFrame, masks: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate the records with recorded data in their window from those without any.

    Args:
        records: Records with ``feature_index``.
        masks: Output of ``read_time_masks``.

    Returns:
        Tuple (records with data, records without data).
    """
    has_data = records_with_data(records, masks)
    return records[has_data], records[~has_data]


def mask_summary(records: pd.DataFrame, masks: np.ndarray, label_column: str) -> pd.DataFrame:
    """Per-class summary of the masks: records, records without data, mean share of data bins.

    Args:
        records: Records with ``feature_index`` and the label.
        masks: Output of ``read_time_masks``.
        label_column: Label column.

    Returns:
        Table indexed by label.
    """
    rows = records[FEATURE_INDEX_COLUMN].to_numpy()
    table = pd.DataFrame(
        {
            label_column: records[label_column].to_numpy(),
            "without_data": ~masks[rows].any(axis=1),
            "data_share": masks[rows].mean(axis=1),
        }
    )
    return table.groupby(label_column).agg(
        records=("without_data", "size"), without_data=("without_data", "sum"), data_share_mean=("data_share", "mean")
    )
