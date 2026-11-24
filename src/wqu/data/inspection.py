"""Inspection of ESM25 waveforms: zero padding, data gaps and spectra.

Every trace is examined for:
- the number of zero samples at the start and at the end of each component
  (front and back padding);
- runs of zeros inside the recorded part (possible data gaps);
- the peak absolute amplitude;
- its Welch power spectrum over the recorded part, averaged per original
  sampling rate (to see whether upsampled records lack high frequencies).

The analysis runs batch by batch, so it scales from the development subset to
the full dataset without loading all traces in memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch

from wqu.data.loading import CV_COMPONENTS

ORIGINAL_SAMPLING_RATE_COLUMN = "original_sampling_rate_hz"
TRACE_START_COLUMN = "trace_start_time"


@dataclass(frozen=True)
class WaveformInspection:
    """Result of the waveform inspection.

    Attributes:
        statistics: One row per record (same index as the inspected records).
        mean_log_psd: Mean log10 power spectral density, frequencies x groups.
        spectrum_counts: Number of component spectra averaged per group.
    """

    statistics: pd.DataFrame
    mean_log_psd: pd.DataFrame
    spectrum_counts: pd.Series


def edge_zero_samples(traces: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Count zero samples at the start and end of each component.

    Args:
        traces: Array of shape (records, components, samples).
        tolerance: Samples with absolute value at or below this are zeros.

    Returns:
        Tuple (leading, trailing), integer arrays of shape (records, components).
        A component made only of zeros has leading = trailing = number of samples.
    """
    nonzero = np.abs(traces) > tolerance
    n_samples = traces.shape[-1]
    has_data = nonzero.any(axis=-1)
    leading = np.where(has_data, nonzero.argmax(axis=-1), n_samples)
    trailing = np.where(has_data, nonzero[..., ::-1].argmax(axis=-1), n_samples)
    return leading, trailing


def internal_zero_runs(
    trace: np.ndarray, leading: int, trailing: int, tolerance: float, min_run_samples: int
) -> tuple[int, int]:
    """Count runs of zeros inside the recorded part of one component.

    Args:
        trace: One component, shape (samples,).
        leading: Zero samples at the start (excluded).
        trailing: Zero samples at the end (excluded).
        tolerance: Zero threshold on the absolute value.
        min_run_samples: Shortest run counted.

    Returns:
        Tuple (number of runs, total samples in these runs).
    """
    recorded = trace[leading : len(trace) - trailing]
    if recorded.size == 0:
        return 0, 0
    is_zero = np.abs(recorded) <= tolerance
    edges = np.diff(np.concatenate(([0], is_zero.astype(np.int8), [0])))
    run_lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    long_runs = run_lengths[run_lengths >= min_run_samples]
    return int(long_runs.size), int(long_runs.sum())


def batch_statistics(
    records: pd.DataFrame, traces: np.ndarray, sampling_rate_hz: float, tolerance: float, min_gap_s: float
) -> pd.DataFrame:
    """Padding, gap and amplitude statistics of a batch of traces.

    Args:
        records: Records of the batch (index reused for the output).
        traces: Array of shape (records, components, samples).
        sampling_rate_hz: Sampling rate of the traces.
        tolerance: Zero threshold on the absolute value.
        min_gap_s: Shortest internal zero run reported as a gap.

    Returns:
        Table with, per component c: ``leading_zero_s_c``, ``trailing_zero_s_c``,
        ``gap_count_c``, ``gap_total_s_c``, ``peak_abs_c``; and per record:
        ``leading_zero_s_max`` (largest front padding), ``trailing_zero_s_min``
        (common back padding) and ``recorded_span_s`` (trace length minus the
        common front and back padding).
    """
    leading, trailing = edge_zero_samples(traces, tolerance)
    min_run_samples = max(1, int(round(min_gap_s * sampling_rate_hz)))
    columns: dict[str, np.ndarray] = {}
    for position, component in enumerate(CV_COMPONENTS):
        gaps = [
            internal_zero_runs(trace[position], lead[position], trail[position], tolerance, min_run_samples)
            for trace, lead, trail in zip(traces, leading, trailing)
        ]
        columns[f"leading_zero_s_{component}"] = leading[:, position] / sampling_rate_hz
        columns[f"trailing_zero_s_{component}"] = trailing[:, position] / sampling_rate_hz
        columns[f"gap_count_{component}"] = np.array([count for count, _ in gaps])
        columns[f"gap_total_s_{component}"] = np.array([total for _, total in gaps]) / sampling_rate_hz
        columns[f"peak_abs_{component}"] = np.abs(traces[:, position]).max(axis=-1)
    statistics = pd.DataFrame(columns, index=records.index)
    trace_length_s = traces.shape[-1] / sampling_rate_hz
    statistics["leading_zero_s_max"] = leading.max(axis=1) / sampling_rate_hz
    statistics["trailing_zero_s_min"] = trailing.min(axis=1) / sampling_rate_hz
    statistics["recorded_span_s"] = (
        trace_length_s - leading.min(axis=1) / sampling_rate_hz - statistics["trailing_zero_s_min"]
    ).clip(lower=0.0)
    return statistics


def component_log_psd(
    trace: np.ndarray, leading: int, trailing: int, sampling_rate_hz: float, segment_samples: int
) -> np.ndarray | None:
    """Log10 Welch power spectral density of the recorded part of one component.

    Args:
        trace: One component, shape (samples,).
        leading: Zero samples at the start (excluded).
        trailing: Zero samples at the end (excluded).
        sampling_rate_hz: Sampling rate.
        segment_samples: Welch segment length.

    Returns:
        Array of ``segment_samples // 2 + 1`` values, or None if the recorded part
        is shorter than one segment or has no energy.
    """
    recorded = trace[leading : len(trace) - trailing].astype(np.float64)
    if recorded.size < segment_samples:
        return None
    _, psd = welch(recorded, fs=sampling_rate_hz, nperseg=segment_samples, detrend="constant")
    positive = psd > 0
    if not positive.any():
        return None
    return np.log10(np.where(positive, psd, psd[positive].min()))


def inspect_waveform_batches(
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    tolerance: float,
    min_gap_s: float,
    segment_s: float,
    spectrum_groups: Sequence[float],
) -> WaveformInspection:
    """Run the inspection over batches of traces in a single pass.

    Args:
        batches: Iterable of (records, traces) as produced by ``iter_waveform_batches``.
        sampling_rate_hz: Sampling rate of the traces.
        tolerance: Zero threshold on the absolute value.
        min_gap_s: Shortest internal zero run reported as a gap.
        segment_s: Welch segment length in seconds.
        spectrum_groups: Original sampling rates for which mean spectra are computed.

    Returns:
        WaveformInspection with statistics and mean spectra.
    """
    segment_samples = int(round(segment_s * sampling_rate_hz))
    frequencies = np.fft.rfftfreq(segment_samples, d=1.0 / sampling_rate_hz)
    sums = {group: np.zeros_like(frequencies) for group in spectrum_groups}
    counts = {group: 0 for group in spectrum_groups}
    tables = []
    for records, traces in batches:
        tables.append(batch_statistics(records, traces, sampling_rate_hz, tolerance, min_gap_s))
        leading, trailing = edge_zero_samples(traces, tolerance)
        groups = records[ORIGINAL_SAMPLING_RATE_COLUMN].to_numpy()
        for trace, lead, trail, group in zip(traces, leading, trailing, groups):
            if group not in sums:
                continue
            for position in range(trace.shape[0]):
                log_psd = component_log_psd(trace[position], lead[position], trail[position], sampling_rate_hz, segment_samples)
                if log_psd is not None:
                    sums[group] += log_psd
                    counts[group] += 1
    mean_log_psd = pd.DataFrame(
        {group: sums[group] / counts[group] for group in spectrum_groups if counts[group] > 0},
        index=pd.Index(frequencies, name="frequency_hz"),
    )
    return WaveformInspection(
        statistics=pd.concat(tables) if tables else pd.DataFrame(),
        mean_log_psd=mean_log_psd,
        spectrum_counts=pd.Series(counts, name="n_spectra"),
    )


def expected_padding_s(records: pd.DataFrame, waveform_length_s: float) -> pd.DataFrame:
    """Front and back padding of each component predicted from the metadata.

    Args:
        records: Metadata rows with parsed start and end times.
        waveform_length_s: Fixed trace length.

    Returns:
        Table with ``expected_leading_s_c`` and ``expected_trailing_s_c`` per component.
    """
    reference = records[TRACE_START_COLUMN]
    columns = {}
    for component in CV_COMPONENTS:
        start_s = (records[f"trace_start_time_orig_cv_{component}"] - reference).dt.total_seconds()
        end_s = (records[f"trace_end_time_orig_cv_{component}"] - reference).dt.total_seconds()
        columns[f"expected_leading_s_{component}"] = start_s.clip(lower=0.0)
        columns[f"expected_trailing_s_{component}"] = (waveform_length_s - end_s).clip(lower=0.0)
    return pd.DataFrame(columns, index=records.index)


def padding_discrepancy_s(statistics: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Measured minus expected padding, per component.

    Args:
        statistics: Output statistics of the inspection.
        expected: Output of ``expected_padding_s`` for the same records.

    Returns:
        Table with ``leading_diff_s_c`` and ``trailing_diff_s_c`` per component.
    """
    columns = {}
    for component in CV_COMPONENTS:
        columns[f"leading_diff_s_{component}"] = (
            statistics[f"leading_zero_s_{component}"] - expected[f"expected_leading_s_{component}"]
        )
        columns[f"trailing_diff_s_{component}"] = (
            statistics[f"trailing_zero_s_{component}"] - expected[f"expected_trailing_s_{component}"]
        )
    return pd.DataFrame(columns, index=statistics.index)


def share_by_label(flags: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Fraction of records with each boolean flag, per label.

    Args:
        flags: Boolean table (records x flags).
        labels: Labels aligned with ``flags``.

    Returns:
        Table (flags x labels) of fractions.
    """
    return flags.groupby(labels).mean().T
