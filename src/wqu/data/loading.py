"""Loading of ESM25 metadata and waveforms.

Metadata can be read in two ways, selected by ``data.source`` in the configuration:

- ``seisbench``: the ESM25 benchmark classes are instantiated (which downloads
  metadata and waveforms into the SeisBench cache on first use) and the
  ``metadata.csv`` file of each dataset is read from the cache folder;
- ``local_csv``: the SeisBench ``metadata.csv`` files are read from a local
  folder, without waveforms (development mode).

In both cases the CSV file is read directly instead of using
``dataset.metadata``, because SeisBench parses it without column types and
turns codes such as ``"00"`` into numbers. The row order is identical, so the
row position (``subset_index``) is also the index of the trace in the
SeisBench dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seisbench.data as sbd

from wqu.config import ColumnsConfig, DataConfig, ProjectConfig, SubsetConfig

CV_COMPONENTS: tuple[str, ...] = ("u", "v", "w")
TRACE_START_COLUMN = "trace_start_time"
ORIGIN_TIME_COLUMN = "source_esm_origin_time"
ORIGINAL_DT_COLUMN = "trace_dt_s_orig_cv_u"
# Rounding to whole hertz removes artefacts of the stored sampling interval (e.g. 127.992 Hz).
SAMPLING_RATE_DECIMALS = 0


def require_columns(metadata: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Raise a KeyError if any of the requested columns is missing.

    Args:
        metadata: Table to check.
        columns: Required column names.
        context: Description used in the error message.
    """
    missing = [column for column in columns if column not in metadata.columns]
    if missing:
        raise KeyError(f"{context}: missing columns {missing}.")


def load_seisbench_dataset(class_name: str, dataset_options: Mapping[str, object]) -> sbd.WaveformDataset:
    """Instantiate a SeisBench benchmark dataset by class name.

    The first call downloads the dataset into the SeisBench cache
    (``SEISBENCH_CACHE_ROOT``); later calls read it from there.

    Args:
        class_name: Name of a class in ``seisbench.data`` (e.g. ``ESM25BadCV``).
        dataset_options: Keyword arguments for the dataset constructor.

    Returns:
        The SeisBench dataset.
    """
    dataset_class = getattr(sbd, class_name, None)
    if dataset_class is None:
        raise ValueError(f"seisbench.data has no dataset class named '{class_name}'.")
    return dataset_class(**dataset_options)


def read_metadata_csv(path: Path, text_columns: Sequence[str]) -> pd.DataFrame:
    """Read a metadata CSV file, keeping code-like columns as text.

    Args:
        path: CSV file.
        text_columns: Columns to read as strings; names absent from the file are ignored.

    Returns:
        Metadata table with a default RangeIndex (row position = trace index).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {column: "string" for column in text_columns if column in header}
    return pd.read_csv(path, dtype=dtypes, low_memory=False)


def parse_datetime_columns(metadata: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert timestamp columns to timezone-aware UTC datetimes.

    Values that cannot be parsed become NaT, so they appear as missing values in
    the exploration instead of stopping the loading.

    Args:
        metadata: Input table.
        columns: Columns to convert; names absent from the table are ignored.

    Returns:
        New table with converted columns.
    """
    parsed = metadata.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column], utc=True, format="ISO8601", errors="coerce")
    return parsed


def add_preferred_magnitude(metadata: pd.DataFrame, priority: Sequence[str]) -> pd.DataFrame:
    """Add the first available magnitude following a priority order.

    Args:
        metadata: Input table.
        priority: Magnitude columns in order of preference.

    Returns:
        New table with ``preferred_magnitude`` (float) and
        ``preferred_magnitude_type`` (name of the column used, missing if none).
    """
    present = [column for column in priority if column in metadata.columns]
    if not present:
        raise KeyError(f"None of the magnitude columns {list(priority)} is present.")
    magnitudes = metadata[present].apply(pd.to_numeric, errors="coerce")
    available = magnitudes.notna()

    enriched = metadata.copy()
    enriched["preferred_magnitude"] = magnitudes.bfill(axis=1).iloc[:, 0]
    magnitude_type = available.idxmax(axis=1).str.replace("source_magnitude_", "", regex=False)
    enriched["preferred_magnitude_type"] = magnitude_type.where(available.any(axis=1)).astype("string")
    return enriched


def _component_columns(template: str, components: Sequence[str] = CV_COMPONENTS) -> list[str]:
    """Expand a column name template over the CV components.

    Args:
        template: Name containing ``{component}``.
        components: Component suffixes.

    Returns:
        List of column names.
    """
    return [template.format(component=component) for component in components]


def add_derived_columns(metadata: pd.DataFrame, waveform_length_s: float, station_id_column: str) -> pd.DataFrame:
    """Add columns derived from the raw recording metadata.

    Added columns:
        - ``original_duration_s``: longest original duration among the CV components;
        - ``recording_span_s``: time from the earliest component start to the latest
          component end, i.e. the part of the 420 s trace that contains data;
        - ``padding_fraction``: fraction of the fixed-length trace filled with zeros;
        - ``component_start_spread_s``: delay between the first and last component start;
        - ``original_sampling_rate_hz``: sampling rate before resampling to 200 Hz;
        - ``origin_year``: year of the event origin time;
        - ``max_unprocessed_pga_cps2``: largest absolute unprocessed PGA among components;
        - station identifier column (``network.station.location``).

    Args:
        metadata: Table with datetime columns already parsed.
        waveform_length_s: Fixed trace length of the dataset.
        station_id_column: Name of the station identifier column to create.

    Returns:
        New table with the derived columns.
    """
    duration_columns = _component_columns("trace_duration_s_orig_cv_{component}")
    start_columns = _component_columns("trace_start_time_orig_cv_{component}")
    end_columns = _component_columns("trace_end_time_orig_cv_{component}")
    pga_columns = _component_columns("trace_{component}_un_pga_cps2")
    station_columns = ["station_network_code", "station_code", "station_location_code"]
    require_columns(
        metadata,
        duration_columns + start_columns + end_columns + pga_columns + station_columns
        + [ORIGINAL_DT_COLUMN, ORIGIN_TIME_COLUMN],
        "add_derived_columns",
    )

    first_start = metadata[start_columns].min(axis=1)
    last_start = metadata[start_columns].max(axis=1)
    last_end = metadata[end_columns].max(axis=1)
    recording_span_s = (last_end - first_start).dt.total_seconds()
    original_dt_s = pd.to_numeric(metadata[ORIGINAL_DT_COLUMN], errors="coerce")

    enriched = metadata.copy()
    enriched["original_duration_s"] = metadata[duration_columns].max(axis=1)
    enriched["recording_span_s"] = recording_span_s
    enriched["padding_fraction"] = (1.0 - recording_span_s / waveform_length_s).clip(lower=0.0, upper=1.0)
    enriched["component_start_spread_s"] = (last_start - first_start).dt.total_seconds()
    enriched["original_sampling_rate_hz"] = (1.0 / original_dt_s.where(original_dt_s > 0)).round(
        SAMPLING_RATE_DECIMALS
    )
    enriched["origin_year"] = metadata[ORIGIN_TIME_COLUMN].dt.year.astype("Int64")
    enriched["max_unprocessed_pga_cps2"] = metadata[pga_columns].abs().max(axis=1)
    enriched[station_id_column] = (
        metadata["station_network_code"].fillna("")
        + "."
        + metadata["station_code"].fillna("")
        + "."
        + metadata["station_location_code"].fillna("")
    ).astype("string")
    return enriched


def prepare_subset_metadata(raw_metadata: pd.DataFrame, data: DataConfig, columns: ColumnsConfig) -> pd.DataFrame:
    """Apply type conversions and derived columns to one subset.

    Args:
        raw_metadata: Metadata as read from the CSV file.
        data: Data configuration.
        columns: Column configuration.

    Returns:
        Prepared metadata table.
    """
    parsed = parse_datetime_columns(raw_metadata, columns.datetime)
    with_magnitude = add_preferred_magnitude(parsed, columns.magnitude_priority)
    return add_derived_columns(with_magnitude, data.waveform_length_s, columns.station_id)


def _metadata_path(subset: SubsetConfig, data: DataConfig) -> Path:
    """Return the metadata CSV path of a subset for the configured data source.

    In ``seisbench`` mode the dataset is instantiated first, which triggers the
    download if the dataset is not yet in the cache.

    Args:
        subset: Subset configuration.
        data: Data configuration.

    Returns:
        Path to the metadata CSV file.
    """
    if data.source == "seisbench":
        dataset = load_seisbench_dataset(subset.seisbench_class, data.dataset_options)
        return Path(dataset.path) / data.seisbench_metadata_file
    return data.local_metadata_dir / subset.local_metadata_file


def load_subset_metadata(config: ProjectConfig) -> dict[str, pd.DataFrame]:
    """Load and prepare the metadata of every configured subset.

    Args:
        config: Project configuration.

    Returns:
        Mapping from subset name to prepared metadata table.
    """
    metadata_by_subset: dict[str, pd.DataFrame] = {}
    for subset in config.data.subsets:
        raw = read_metadata_csv(_metadata_path(subset, config.data), config.columns.text)
        metadata_by_subset[subset.name] = prepare_subset_metadata(raw, config.data, config.columns)
    return metadata_by_subset


def combine_subset_metadata(
    metadata_by_subset: Mapping[str, pd.DataFrame],
    subsets: Sequence[SubsetConfig],
    columns: ColumnsConfig,
) -> pd.DataFrame:
    """Stack the subsets into one table with label, subset name and trace index.

    Columns present in only one subset are kept and are missing for the others.

    Args:
        metadata_by_subset: Prepared metadata per subset.
        subsets: Subset configurations (source of the labels).
        columns: Column configuration (names of the added columns).

    Returns:
        Combined table with a new RangeIndex.
    """
    frames = []
    for subset in subsets:
        if subset.name not in metadata_by_subset:
            raise KeyError(f"No metadata loaded for subset '{subset.name}'.")
        frame = metadata_by_subset[subset.name].copy()
        frame[columns.subset_index] = np.arange(len(frame))
        frame[columns.subset] = subset.name
        frame[columns.label] = subset.label
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined[columns.subset] = combined[columns.subset].astype("category")
    return combined


def load_esm25_metadata(config: ProjectConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the ESM25 metadata of all subsets and combine them.

    Args:
        config: Project configuration.

    Returns:
        Tuple of (metadata per subset, combined metadata).
    """
    metadata_by_subset = load_subset_metadata(config)
    combined = combine_subset_metadata(metadata_by_subset, config.data.subsets, config.columns)
    return metadata_by_subset, combined


def load_seisbench_datasets(config: ProjectConfig) -> dict[str, sbd.WaveformDataset]:
    """Instantiate the SeisBench dataset of every configured subset.

    Args:
        config: Project configuration (``data.source`` must be ``seisbench``).

    Returns:
        Mapping from subset name to SeisBench dataset.
    """
    if config.data.source != "seisbench":
        raise ValueError("Waveforms are only available with data.source = 'seisbench'.")
    return {
        subset.name: load_seisbench_dataset(subset.seisbench_class, config.data.dataset_options)
        for subset in config.data.subsets
    }


def signal_windows_s(record: pd.Series, components: Sequence[str] = CV_COMPONENTS) -> dict[str, tuple[float, float]]:
    """Return the time interval containing recorded data for each component.

    Times are in seconds from ``trace_start_time``, the first sample of the
    fixed-length trace. Samples outside these intervals are zero padding.

    Args:
        record: Row of the metadata with parsed datetime columns.
        components: Component suffixes.

    Returns:
        Mapping from component to (start_s, end_s).
    """
    reference = record[TRACE_START_COLUMN]
    windows = {}
    for component in components:
        start = (record[f"trace_start_time_orig_cv_{component}"] - reference).total_seconds()
        end = (record[f"trace_end_time_orig_cv_{component}"] - reference).total_seconds()
        windows[component] = (start, end)
    return windows
