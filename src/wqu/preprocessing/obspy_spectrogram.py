"""Spectrogram images in the style of Shalabi et al. (2026), from the stored traces.

Shalabi et al. draw the spectrogram of each component with ``obspy.Trace.spectrogram`` (default
settings), crop the axes from the figure, reduce it to 64 x 64 pixels and divide by 255. The
images of this module follow the same steps on the arrays, without drawing a figure:

1. each component is cut to its own recorded samples (the stored traces are zero-padded to a fixed
   length; the ESM records used in the paper are not), or, with the ``window`` time axis, to the
   window of ``preprocessing.window`` (the same for the three components, padding included);
2. the segment is resampled to the original sampling rate of the record (the stored traces are all
   at the same rate), at most the stored one;
3. the mean is removed and the short-time Fourier transform is computed as ObsPy does: Hann window
   of ``window_samples`` samples, ``overlap`` between windows, each window zero-padded to
   ``pad_factor * window_samples`` samples, power spectral density without the zero frequency;
4. the power is converted to decibels (or to its square root, the amplitude drawn by ObsPy by
   default) and scaled to [0, 1] by its minimum and maximum, as the colour scale of the figure does;
5. the scaled values are drawn with the colour scale of ObsPy (viridis) and converted to grey levels
   with the ITU-R 601 luminance (0.299 R + 0.587 G + 0.114 B, the conversion of PIL and torchvision):
   a monotonic but not linear map from [0, 1] to about [0.12, 0.85];
6. the image is reduced to ``n_frequencies x n_times`` by averaging over the area of each output
   cell, as a reduction of the drawn figure does (the luminance is linear in the colours, so it can
   be taken before the reduction).

A component without recorded data, or too short for two windows, gives a constant image: the grey
level of the lowest colour (the figure of a constant spectrogram has a single colour), or zeros
without the grey-level conversion. The 8-bit quantization of the figure is not reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pandas as pd
from matplotlib import colormaps
from scipy.signal import resample_poly, spectrogram

from wqu.config import (
    DECIBEL_AMPLITUDE,
    LOGARITHMIC_FREQUENCY_AXIS,
    MIN_MAX_NORMALIZATION,
    VIRIDIS_LUMINANCE_GRAYSCALE,
    ORIGINAL_SAMPLING_RATE,
    RECORDED_TIME_AXIS,
    ObspySpectrogramConfig,
    PreprocessingConfig,
)
from wqu.data.inspection import edge_zero_samples
from wqu.preprocessing.pipeline import locate_window
from wqu.preprocessing.windowing import extract_segment

ORIGINAL_SAMPLING_RATE_COLUMN = "original_sampling_rate_hz"
RESAMPLING_MAX_DENOMINATOR = 1000
MIN_WINDOWS = 2
GRAYSCALE_COLORMAP = "viridis"
LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114])


@dataclass(frozen=True)
class ObspyImage:
    """Image of one record and what it was computed from.

    Attributes:
        image: Array of shape (components, n_frequencies, n_times), float32.
        sampling_rate_hz: Sampling rate of the transformed segments.
        durations_s: Duration of the segment of each component (0 without data).
        has_data: Whether each component had recorded samples, enough for two windows.
    """

    image: np.ndarray
    sampling_rate_hz: float
    durations_s: np.ndarray
    has_data: np.ndarray


def component_spans(trace: np.ndarray, tolerance: float) -> list[tuple[int, int]]:
    """Recorded samples of each component, from the first to the last non-zero one.

    Args:
        trace: Stored trace, shape (components, samples).
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        One (start, stop) pair per component; empty (start >= stop) for a component without data.
    """
    leading, trailing = edge_zero_samples(trace[None], tolerance)
    n_samples = trace.shape[-1]
    return [(int(lead), n_samples - int(trail)) for lead, trail in zip(leading[0], trailing[0])]


def record_sampling_rate(record: pd.Series, stored_rate_hz: float, setting: str) -> float:
    """Sampling rate at which a record is transformed.

    Args:
        record: Metadata row of the record (with ``original_sampling_rate_hz``).
        stored_rate_hz: Sampling rate of the stored traces.
        setting: ``original`` or ``stored`` (``ObspySpectrogramConfig.sampling_rate``).

    Returns:
        The original rate of the record, at most the stored one, or the stored rate.

    Raises:
        ValueError: If the original rate is requested but missing or not positive.
    """
    if setting != ORIGINAL_SAMPLING_RATE:
        return stored_rate_hz
    original = float(record[ORIGINAL_SAMPLING_RATE_COLUMN])
    if not original > 0:
        raise ValueError(f"Missing or invalid {ORIGINAL_SAMPLING_RATE_COLUMN}: {record[ORIGINAL_SAMPLING_RATE_COLUMN]}.")
    return min(original, stored_rate_hz)


def resample_to_rate(segment: np.ndarray, rate_hz: float, target_rate_hz: float) -> np.ndarray:
    """Resample a one-dimensional segment with a polyphase anti-alias filter.

    Args:
        segment: Samples at ``rate_hz``.
        rate_hz: Current sampling rate.
        target_rate_hz: Output sampling rate.

    Returns:
        Resampled samples (the input itself when the rates are equal).
    """
    ratio = Fraction(target_rate_hz / rate_hz).limit_denominator(RESAMPLING_MAX_DENOMINATOR)
    if ratio == 1:
        return segment
    return resample_poly(segment, ratio.numerator, ratio.denominator)


def obspy_power(samples: np.ndarray, rate_hz: float, settings: ObspySpectrogramConfig) -> tuple[np.ndarray, np.ndarray] | None:
    """Power spectral density as computed by ``obspy.Trace.spectrogram``.

    Args:
        samples: One-dimensional segment.
        rate_hz: Sampling rate of the segment.
        settings: Image settings (window, overlap, zero-padding).

    Returns:
        Tuple (power of shape (frequencies, windows), frequencies in Hz) without the zero frequency,
        or None if the segment has fewer than ``MIN_WINDOWS`` windows (ObsPy refuses it).
    """
    if len(samples) < settings.window_samples:
        return None
    frequencies, _, power = spectrogram(
        samples - samples.mean(),
        fs=rate_hz,
        window=np.hanning(settings.window_samples),
        nperseg=settings.window_samples,
        noverlap=int(settings.window_samples * settings.overlap),
        nfft=settings.pad_factor * settings.window_samples,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    if power.shape[1] < MIN_WINDOWS:
        return None
    return power[1:], frequencies[1:]


def amplitude_values(power: np.ndarray, scale: str, floor_power: float) -> np.ndarray:
    """Values drawn for a power array.

    Args:
        power: Power spectral density.
        scale: ``decibel`` (10 log10 of the power, floored) or ``linear`` (square root of the power).
        floor_power: Smallest power used in the logarithm.

    Returns:
        Array of the same shape.
    """
    if scale == DECIBEL_AMPLITUDE:
        return 10.0 * np.log10(np.maximum(power, floor_power))
    return np.sqrt(power)


def logarithmic_rows(values: np.ndarray, frequencies_hz: np.ndarray, target_hz: np.ndarray) -> np.ndarray:
    """Values at logarithmically spaced frequencies, by linear interpolation in log frequency.

    Target frequencies outside the computed ones take the value of the nearest computed frequency.

    Args:
        values: Array of shape (frequencies, windows).
        frequencies_hz: Increasing frequencies of the rows of ``values``.
        target_hz: Target frequencies.

    Returns:
        Array of shape (len(target_hz), windows).
    """
    log_source, log_target = np.log10(frequencies_hz), np.log10(target_hz)
    return np.stack([np.interp(log_target, log_source, column) for column in values.T], axis=1)


def min_max_scaled(values: np.ndarray) -> np.ndarray:
    """Values scaled to [0, 1] by their minimum and maximum; zeros for a constant array.

    Args:
        values: Array.

    Returns:
        Array of the same shape.
    """
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


@lru_cache(maxsize=1)
def luminance_table() -> np.ndarray:
    """Grey level of every colour of the viridis colour scale (ITU-R 601 luminance).

    Returns:
        Array with one grey level per colour of the scale, from the lowest value to the highest.
    """
    colormap = colormaps[GRAYSCALE_COLORMAP]
    return colormap(np.arange(colormap.N))[:, :3] @ LUMINANCE_WEIGHTS


def grey_levels(values: np.ndarray, grayscale: str) -> np.ndarray:
    """Grey levels of values in [0, 1], as in the figure converted to grey levels.

    Each value picks a colour of the scale as Matplotlib does (``N`` equal intervals of [0, 1], the
    value 1 in the last one), whose grey level is read from ``luminance_table``.

    Args:
        values: Array of values in [0, 1].
        grayscale: ``viridis_luminance`` (viridis colours, ITU-R 601 luminance) or ``none`` (values kept).

    Returns:
        Array of the same shape.
    """
    if grayscale != VIRIDIS_LUMINANCE_GRAYSCALE:
        return values
    table = luminance_table()
    indices = np.clip((np.asarray(values) * len(table)).astype(np.int64), 0, len(table) - 1)
    return table[indices]


def area_weights(n_input: int, n_output: int) -> np.ndarray:
    """Weights that average input cells over the area of each output cell along one axis.

    The axis is divided into ``n_input`` equal input cells and ``n_output`` equal output cells;
    each output value is the mean of the input values weighted by their overlap with the output cell.

    Args:
        n_input: Input cells.
        n_output: Output cells.

    Returns:
        Array of shape (n_output, n_input) whose rows sum to one.
    """
    input_edges = np.linspace(0.0, 1.0, n_input + 1)
    output_edges = np.linspace(0.0, 1.0, n_output + 1)
    overlap = np.clip(
        np.minimum(output_edges[1:, None], input_edges[None, 1:]) - np.maximum(output_edges[:-1, None], input_edges[None, :-1]),
        0.0,
        None,
    )
    return overlap / overlap.sum(axis=1, keepdims=True)


def resized(image: np.ndarray, n_rows: int, n_columns: int) -> np.ndarray:
    """Image reduced (or enlarged) to a fixed size by area averaging.

    Args:
        image: Array of shape (rows, columns).
        n_rows: Output rows.
        n_columns: Output columns.

    Returns:
        Array of shape (n_rows, n_columns).
    """
    return area_weights(image.shape[0], n_rows) @ image @ area_weights(image.shape[1], n_columns).T


def component_image(
    samples: np.ndarray, rate_hz: float, settings: ObspySpectrogramConfig, floor_power: float
) -> np.ndarray | None:
    """Image of one component segment.

    Args:
        samples: Segment at ``rate_hz``.
        rate_hz: Sampling rate of the segment.
        settings: Image settings.
        floor_power: Smallest power used in the decibel conversion.

    Returns:
        Array of shape (n_frequencies, n_times), or None if the segment is too short for two windows.
    """
    transformed = obspy_power(samples, rate_hz, settings)
    if transformed is None:
        return None
    power, frequencies_hz = transformed
    values = amplitude_values(power, settings.amplitude_scale, floor_power)
    if settings.frequency_axis == LOGARITHMIC_FREQUENCY_AXIS:
        target_hz = np.geomspace(settings.min_frequency_hz, settings.max_frequency_hz, settings.n_frequencies)
        values = logarithmic_rows(values, frequencies_hz, target_hz)
    if settings.normalization == MIN_MAX_NORMALIZATION:
        values = grey_levels(min_max_scaled(values), settings.grayscale)
    return resized(values, settings.n_frequencies, settings.n_times)


def component_segments(
    trace: np.ndarray, record: pd.Series, stored_rate_hz: float, preprocessing: PreprocessingConfig, tolerance: float
) -> list[np.ndarray]:
    """Segment of each component at the stored sampling rate.

    Args:
        trace: Stored trace, shape (components, samples).
        record: Metadata row of the record (used by the anchoring of the ``window`` time axis).
        stored_rate_hz: Sampling rate of the stored traces.
        preprocessing: Preprocessing configuration with the ``obspy_spectrogram`` settings.
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        One array per component (empty for a component without data with the ``recorded`` time axis).
    """
    settings = preprocessing.obspy_spectrogram
    if settings.time_axis == RECORDED_TIME_AXIS:
        return [trace[component, start:stop].astype(np.float64) for component, (start, stop)
                in enumerate(component_spans(trace, tolerance))]
    centred, bounds, _ = locate_window(trace, record, stored_rate_hz, preprocessing, tolerance)
    return list(extract_segment(centred, bounds.window_start, bounds.window_stop))


def record_image(
    trace: np.ndarray, record: pd.Series, stored_rate_hz: float, preprocessing: PreprocessingConfig, tolerance: float
) -> ObspyImage:
    """Image of the three components of one record.

    Args:
        trace: Stored trace, shape (components, samples).
        record: Metadata row of the record.
        stored_rate_hz: Sampling rate of the stored traces.
        preprocessing: Preprocessing configuration with the ``obspy_spectrogram`` settings.
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        ObspyImage.

    Raises:
        ValueError: If the configuration has no ``obspy_spectrogram`` settings.
    """
    settings = preprocessing.obspy_spectrogram
    if settings is None:
        raise ValueError("The configuration has no preprocessing.obspy_spectrogram settings.")
    rate_hz = record_sampling_rate(record, stored_rate_hz, settings.sampling_rate)
    empty_value = (float(grey_levels(np.zeros(1), settings.grayscale)[0])
                   if settings.normalization == MIN_MAX_NORMALIZATION else 0.0)
    image = np.full((trace.shape[0], settings.n_frequencies, settings.n_times), empty_value, dtype=np.float32)
    durations_s = np.zeros(trace.shape[0])
    has_data = np.zeros(trace.shape[0], dtype=bool)
    for component, segment in enumerate(component_segments(trace, record, stored_rate_hz, preprocessing, tolerance)):
        durations_s[component] = len(segment) / stored_rate_hz
        if not np.any(np.abs(segment) > tolerance):
            continue
        values = component_image(resample_to_rate(segment, stored_rate_hz, rate_hz), rate_hz, settings,
                                 preprocessing.floor_power)
        if values is not None:
            image[component] = values
            has_data[component] = True
    return ObspyImage(image=image, sampling_rate_hz=rate_hz, durations_s=durations_s, has_data=has_data)


def image_axes(settings: ObspySpectrogramConfig) -> tuple[np.ndarray, np.ndarray]:
    """Centres of the rows and columns of the stored images.

    With the linear frequency axis the rows are in fractions of the Nyquist frequency of each record
    (records at different sampling rates share the rows, not the frequencies); with the logarithmic
    axis they are in Hz. The columns are in fractions of the segment of each component.

    Args:
        settings: Image settings.

    Returns:
        Tuple (row centres, column centres), both increasing and positive.
    """
    if settings.frequency_axis == LOGARITHMIC_FREQUENCY_AXIS:
        rows = np.geomspace(settings.min_frequency_hz, settings.max_frequency_hz, settings.n_frequencies)
    else:
        rows = (np.arange(settings.n_frequencies) + 0.5) / settings.n_frequencies
    return rows, (np.arange(settings.n_times) + 0.5) / settings.n_times
