"""From stored traces to spectrogram and scalogram arrays.

``process_record`` chains the windowing and the transforms for one record;
``window_table`` and ``preprocess_and_store`` apply them to batches of records,
so the same code runs on the development subset and on the full dataset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wqu.config import PreprocessingConfig
from wqu.data.loading import CV_COMPONENTS, read_metadata_csv
from wqu.preprocessing.anchoring import window_anchor
from wqu.preprocessing.transform import (
    TimeFrequencyGrid,
    build_grid,
    grid_from_axes,
    power_to_db,
    scalogram,
    spectrogram,
)
from wqu.preprocessing.windowing import (
    WindowBounds,
    extract_segment,
    prepare_trace,
    resample_segment,
    taper_segment,
    window_bounds,
    window_cut_statistics,
)

REPRESENTATIONS: tuple[str, ...] = ("spectrogram", "scalogram")
FEATURE_INDEX_COLUMN = "feature_index"
FEATURE_DTYPE = "float16"


@dataclass(frozen=True)
class ProcessedRecord:
    """Result of the preprocessing of one record.

    Attributes:
        bounds: Analysis window in samples of the stored trace.
        statistics: Window statistics (see ``window_cut_statistics``).
        window_waveform: Resampled waveform of the analysis window, shape (components, samples).
        power: Power per representation, each of shape (components, frequencies, times).
    """

    bounds: WindowBounds
    statistics: dict[str, float]
    window_waveform: np.ndarray
    power: dict[str, np.ndarray]


def grid_from_config(config: PreprocessingConfig) -> TimeFrequencyGrid:
    """Build the common time-frequency grid from the configuration.

    Args:
        config: Preprocessing configuration.

    Returns:
        TimeFrequencyGrid.
    """
    return build_grid(
        config.min_frequency_hz, config.max_frequency_hz, config.n_frequencies, config.window_length_s, config.n_times
    )


def locate_window(
    trace: np.ndarray, record: pd.Series, sampling_rate_hz: float, config: PreprocessingConfig, tolerance: float
) -> tuple[np.ndarray, WindowBounds, dict[str, float]]:
    """Remove the mean of the recorded samples and locate the analysis window around its anchor.

    Args:
        trace: Stored trace, shape (components, samples).
        record: Metadata row of the record (used by the metadata-based anchoring methods).
        sampling_rate_hz: Sampling rate of the stored trace.
        config: Preprocessing configuration.
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        Tuple (mean-removed trace, window bounds, window statistics).
    """
    prepared = prepare_trace(trace, tolerance)
    anchor = window_anchor(prepared, record, sampling_rate_hz, config.anchor_method, config.anchoring)
    bounds = window_bounds(
        anchor, sampling_rate_hz, config.pre_anchor_s, config.post_anchor_s, config.context_margin_s
    )
    return prepared.centred, bounds, window_cut_statistics(prepared.centred, bounds, tolerance, sampling_rate_hz)


def process_record(
    trace: np.ndarray,
    record: pd.Series,
    sampling_rate_hz: float,
    config: PreprocessingConfig,
    grid: TimeFrequencyGrid,
    tolerance: float,
) -> ProcessedRecord:
    """Window, condition and transform one record.

    Args:
        trace: Stored trace, shape (components, samples).
        record: Metadata row of the record.
        sampling_rate_hz: Sampling rate of the stored trace.
        config: Preprocessing configuration.
        grid: Common time-frequency grid.
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        ProcessedRecord with power arrays (not yet in decibels).
    """
    centred, bounds, statistics = locate_window(trace, record, sampling_rate_hz, config, tolerance)
    segment = taper_segment(extract_segment(centred, bounds.segment_start, bounds.segment_stop), config.taper_fraction)
    resampled = resample_segment(
        segment, sampling_rate_hz, config.target_sampling_rate_hz, config.antialias_corner_hz, config.antialias_order
    )
    rate = config.target_sampling_rate_hz
    margin_samples = int(round(config.context_margin_s * rate))
    window_samples = int(round(config.window_length_s * rate))
    power = {
        "spectrogram": spectrogram(
            resampled, rate, grid, config.context_margin_s, config.stft_window_s, config.stft_window_type
        ),
        "scalogram": scalogram(resampled, rate, grid, config.context_margin_s, config.window_length_s, config.morlet_w0),
    }
    return ProcessedRecord(
        bounds=bounds,
        statistics=statistics,
        window_waveform=resampled[:, margin_samples : margin_samples + window_samples],
        power=power,
    )


def window_table(
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    config: PreprocessingConfig,
    tolerance: float,
) -> pd.DataFrame:
    """Window statistics of every record, without computing the transforms.

    Args:
        batches: Iterable of (records, traces).
        sampling_rate_hz: Sampling rate of the stored traces.
        config: Preprocessing configuration.
        tolerance: Zero threshold.

    Returns:
        Table indexed like the records, with window bounds (samples) and statistics.
    """
    rows = {}
    for records, traces in batches:
        for (index, record), trace in zip(records.iterrows(), traces):
            _, bounds, statistics = locate_window(trace, record, sampling_rate_hz, config, tolerance)
            rows[index] = {**asdict(bounds), **statistics}
    return pd.DataFrame.from_dict(rows, orient="index")


def pooled_power_values(
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    config: PreprocessingConfig,
    tolerance: float,
) -> dict[str, np.ndarray]:
    """Pool all power values of the given records, per representation.

    Used to inspect the distributions before choosing the decibel floor.

    Args:
        batches: Iterable of (records, traces).
        sampling_rate_hz: Sampling rate of the stored traces.
        config: Preprocessing configuration.
        tolerance: Zero threshold.

    Returns:
        Mapping from representation to a flat array of power values.
    """
    grid = grid_from_config(config)
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
    for records, traces in batches:
        for (_, record), trace in zip(records.iterrows(), traces):
            processed = process_record(trace, record, sampling_rate_hz, config, grid, tolerance)
            for name in REPRESENTATIONS:
                pooled[name].append(processed.power[name].ravel())
    return {name: np.concatenate(values) for name, values in pooled.items()}


def feature_file_path(config: PreprocessingConfig, source: str) -> Path:
    """Path of the feature file produced from a waveform source.

    Args:
        config: Preprocessing configuration.
        source: Waveform source name (e.g. ``dev_subset``).

    Returns:
        Path of the HDF5 file.
    """
    return config.output_dir / source / "features.h5"


def preprocess_and_store(
    records: pd.DataFrame,
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    config: PreprocessingConfig,
    tolerance: float,
    output_path: Path,
) -> pd.DataFrame:
    """Compute both representations of every record and store them in decibels.

    The HDF5 file contains one dataset per representation of shape
    (records, components, frequencies, times) in float16, the grid
    (``frequencies_hz``, ``times_s``) and the preprocessing parameters as
    attributes. Row ``i`` corresponds to the record with ``feature_index == i``.
    The file is written under a temporary name and renamed when complete.

    Args:
        records: Records to process (their index is used to match the batches).
        batches: Iterable of (records, traces) covering ``records``.
        sampling_rate_hz: Sampling rate of the stored traces.
        config: Preprocessing configuration.
        tolerance: Zero threshold.
        output_path: HDF5 file to create.

    Returns:
        Table of the stored records with ``feature_index``, window bounds and statistics;
        also written next to the HDF5 file as ``records.csv``.
    """
    grid = grid_from_config(config)
    positions = pd.Series(np.arange(len(records)), index=records.index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".partial.h5")
    shape = (len(records), len(CV_COMPONENTS), len(grid.frequencies_hz), len(grid.times_s))
    rows = {}
    with h5py.File(temporary_path, mode="w") as output:
        datasets = {
            name: output.create_dataset(name, shape=shape, dtype=FEATURE_DTYPE, chunks=(1, *shape[1:]))
            for name in REPRESENTATIONS
        }
        output.create_dataset("frequencies_hz", data=grid.frequencies_hz)
        output.create_dataset("times_s", data=grid.times_s)
        for key, value in asdict(config).items():
            if isinstance(value, (bool, int, float, str)):
                output.attrs[key] = value
        with tqdm(total=len(records), desc="preprocessing") as progress:
            for batch, traces in batches:
                for (index, record), trace in zip(batch.iterrows(), traces):
                    processed = process_record(trace, record, sampling_rate_hz, config, grid, tolerance)
                    row = int(positions[index])
                    for name in REPRESENTATIONS:
                        datasets[name][row] = power_to_db(
                            processed.power[name], config.reference_power, config.floor_power
                        ).astype(FEATURE_DTYPE)
                    rows[index] = {FEATURE_INDEX_COLUMN: row, **asdict(processed.bounds), **processed.statistics}
                    progress.update(1)
    temporary_path.replace(output_path)
    table = records.join(pd.DataFrame.from_dict(rows, orient="index"))
    table.to_csv(feature_records_path(output_path))
    return table


def feature_records_path(features_path: Path) -> Path:
    """Path of the record table written next to a feature file.

    Args:
        features_path: Feature file written by ``preprocess_and_store``.

    Returns:
        Path of ``records.csv``.
    """
    return features_path.with_name("records.csv")


def read_feature_records(features_path: Path, text_columns: Sequence[str]) -> pd.DataFrame:
    """Read the record table of a feature file with code-like columns as text.

    Args:
        features_path: Feature file written by ``preprocess_and_store``.
        text_columns: Columns to read as strings (``columns.text`` of the configuration).

    Returns:
        Table indexed like the metadata of the records, with ``feature_index`` (defragmented copy,
        so that columns can be added without performance warnings).
    """
    table = read_metadata_csv(feature_records_path(features_path), text_columns)
    return table.set_index(table.columns[0]).rename_axis(None).copy()


def read_features(path: Path, representation: str, rows: np.ndarray) -> tuple[np.ndarray, TimeFrequencyGrid]:
    """Read stored decibel arrays and the grid.

    Args:
        path: Feature file written by ``preprocess_and_store``.
        representation: ``spectrogram`` or ``scalogram``.
        rows: Rows to read (``feature_index`` values).

    Returns:
        Tuple (array of shape (rows, components, frequencies, times) in float32, grid).
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation '{representation}'; expected one of {REPRESENTATIONS}.")
    order = np.argsort(rows)
    with h5py.File(path, mode="r") as stored:
        values = stored[representation][np.asarray(rows)[order]].astype(np.float32)
        frequencies = stored["frequencies_hz"][()]
        times = stored["times_s"][()]
    restored = np.empty_like(values)
    restored[order] = values
    return restored, grid_from_axes(frequencies, times)
