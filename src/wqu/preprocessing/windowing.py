"""Selection and conditioning of the analysis window of each record.

Steps, all applied identically to the three components of a record:
1. remove the mean of each component over its recorded samples (zero padding stays zero);
2. find the time of the peak absolute acceleration over the three components (PGA);
3. take a window from ``pre_pga_s`` before to ``post_pga_s`` after the PGA, extended by a
   context margin on both sides (samples outside the stored trace are zeros);
4. taper the ends of the extended segment;
5. low-pass filter and resample it.

The window is the same for the three components, so they stay aligned in time.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.signal.windows import tukey

from wqu.data.loading import CV_COMPONENTS

RESAMPLING_MAX_DENOMINATOR = 1000


@dataclass(frozen=True)
class WindowBounds:
    """Sample indices of the analysis window in the stored trace.

    Indices may fall outside the stored trace when the PGA is close to its ends;
    the missing samples are filled with zeros.

    Attributes:
        pga_sample: Sample of the peak absolute acceleration.
        window_start: First sample of the analysis window.
        window_stop: Sample after the last one of the analysis window.
        segment_start: First sample of the window extended by the context margin.
        segment_stop: Sample after the last one of the extended window.
    """

    pga_sample: int
    window_start: int
    window_stop: int
    segment_start: int
    segment_stop: int


def remove_recorded_mean(traces: np.ndarray, leading: np.ndarray, trailing: np.ndarray) -> np.ndarray:
    """Subtract from each component the mean of its recorded samples.

    Args:
        traces: Array of shape (components, samples).
        leading: Zero samples at the start of each component.
        trailing: Zero samples at the end of each component.

    Returns:
        New array; zero padding is left unchanged.
    """
    centred = traces.astype(np.float64, copy=True)
    n_samples = traces.shape[-1]
    for component, (lead, trail) in enumerate(zip(leading, trailing)):
        stop = n_samples - int(trail)
        if stop > lead:
            centred[component, lead:stop] -= centred[component, lead:stop].mean()
    return centred


def pga_sample(traces: np.ndarray) -> int:
    """Sample index of the largest absolute value over all components.

    Args:
        traces: Array of shape (components, samples).

    Returns:
        Sample index.
    """
    return int(np.unravel_index(np.argmax(np.abs(traces)), traces.shape)[1])


def window_bounds(
    pga_index: int, sampling_rate_hz: float, pre_pga_s: float, post_pga_s: float, context_margin_s: float
) -> WindowBounds:
    """Compute the analysis window and its extension around the PGA sample.

    Args:
        pga_index: Sample of the PGA.
        sampling_rate_hz: Sampling rate of the stored trace.
        pre_pga_s: Window length before the PGA.
        post_pga_s: Window length after the PGA.
        context_margin_s: Extension on each side.

    Returns:
        WindowBounds.
    """
    window_start = pga_index - int(round(pre_pga_s * sampling_rate_hz))
    window_stop = pga_index + int(round(post_pga_s * sampling_rate_hz))
    margin = int(round(context_margin_s * sampling_rate_hz))
    return WindowBounds(pga_index, window_start, window_stop, window_start - margin, window_stop + margin)


def extract_segment(traces: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Copy samples ``start:stop`` of every component, with zeros outside the trace.

    Args:
        traces: Array of shape (components, samples).
        start: First sample (may be negative).
        stop: Sample after the last one (may exceed the trace length).

    Returns:
        Array of shape (components, stop - start).
    """
    n_samples = traces.shape[-1]
    segment = np.zeros((traces.shape[0], stop - start), dtype=np.float64)
    source_start, source_stop = max(start, 0), min(stop, n_samples)
    if source_stop > source_start:
        segment[:, source_start - start : source_stop - start] = traces[:, source_start:source_stop]
    return segment


def taper_segment(segment: np.ndarray, taper_fraction: float) -> np.ndarray:
    """Apply a Tukey taper to both ends of every component.

    Args:
        segment: Array of shape (components, samples).
        taper_fraction: Fraction of the length tapered at each end.

    Returns:
        Tapered array.
    """
    return segment * tukey(segment.shape[-1], alpha=2.0 * taper_fraction)


def resample_segment(
    segment: np.ndarray, sampling_rate_hz: float, target_rate_hz: float, corner_hz: float, order: int
) -> np.ndarray:
    """Low-pass filter (zero phase) and resample every component.

    Integer decimation keeps every n-th sample after the low-pass; other ratios
    use polyphase resampling.

    Args:
        segment: Array of shape (components, samples).
        sampling_rate_hz: Current sampling rate.
        target_rate_hz: Output sampling rate.
        corner_hz: Corner frequency of the Butterworth low-pass.
        order: Order of the Butterworth filter.

    Returns:
        Resampled array of shape (components, new samples).
    """
    sos = butter(order, corner_hz, btype="lowpass", fs=sampling_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, segment, axis=-1)
    ratio = Fraction(target_rate_hz / sampling_rate_hz).limit_denominator(RESAMPLING_MAX_DENOMINATOR)
    if ratio == 1:
        return filtered
    if ratio.numerator == 1:
        return filtered[..., :: ratio.denominator]
    return resample_poly(filtered, ratio.numerator, ratio.denominator, axis=-1)


def window_cut_statistics(
    traces: np.ndarray, bounds: WindowBounds, tolerance: float, sampling_rate_hz: float
) -> dict[str, float]:
    """Describe what the analysis window keeps and removes from a record.

    Args:
        traces: Mean-removed array of shape (components, samples).
        bounds: Window of the record.
        tolerance: Zero threshold on the absolute value.
        sampling_rate_hz: Sampling rate of the stored trace.

    Returns:
        Dictionary with, per component c: ``nonzero_cut_before_c`` and
        ``nonzero_cut_after_c`` (recorded samples removed before and after the
        window) and ``missing_in_window_s_c`` (seconds of the window without data);
        and per record: ``pga_time_s`` (from the trace start), ``recorded_fraction_kept``
        and ``energy_fraction_kept`` (share of recorded samples and of the sum of
        squared amplitudes inside the window).
    """
    n_samples = traces.shape[-1]
    start, stop = max(bounds.window_start, 0), min(bounds.window_stop, n_samples)
    nonzero = np.abs(traces) > tolerance
    energy = traces**2
    statistics: dict[str, float] = {"pga_time_s": bounds.pga_sample / sampling_rate_hz}
    window_samples = bounds.window_stop - bounds.window_start
    for position, component in enumerate(CV_COMPONENTS):
        statistics[f"nonzero_cut_before_{component}"] = float(nonzero[position, :start].sum())
        statistics[f"nonzero_cut_after_{component}"] = float(nonzero[position, stop:].sum())
        kept = float(nonzero[position, start:stop].sum())
        statistics[f"missing_in_window_s_{component}"] = (window_samples - kept) / sampling_rate_hz
    total_recorded = nonzero.sum()
    total_energy = energy.sum()
    statistics["recorded_fraction_kept"] = float(nonzero[:, start:stop].sum() / total_recorded) if total_recorded else 0.0
    statistics["energy_fraction_kept"] = float(energy[:, start:stop].sum() / total_energy) if total_energy else 0.0
    return statistics
