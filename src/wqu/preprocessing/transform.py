"""Time-frequency representations on a common logarithmic grid.

Both representations return power on the same grid of ``n_frequencies``
logarithmically spaced frequencies and ``n_times`` time bins covering the
analysis window:

- spectrogram: short-time Fourier transform with one window length for all
  frequencies; one frame centred on each time bin; the power spectral density
  of the linear frequency bins is averaged within logarithmic bands (or
  interpolated where a band contains no bin);
- scalogram: continuous wavelet transform with a Morlet wavelet, whose duration
  shrinks with frequency; the power is averaged within each time bin. The
  transform follows the definition of ``obspy.signal.tf_misfit.cwt``
  (same wavelet, normalization and frequencies) but is computed in the
  frequency domain with the analytic Fourier transform of the wavelet, which
  is faster and avoids sampling a very short wavelet at high frequencies.

The transforms are computed on the window extended by a context margin, and
only the window is kept, so that edge effects fall outside the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as scipy_fft
from scipy.signal import get_window


@dataclass(frozen=True)
class TimeFrequencyGrid:
    """Common grid of the representations.

    Attributes:
        frequencies_hz: Centre frequencies (logarithmically spaced).
        frequency_edges_hz: Band edges (geometric midpoints between centres).
        times_s: Centres of the time bins, in seconds from the window start.
    """

    frequencies_hz: np.ndarray
    frequency_edges_hz: np.ndarray
    times_s: np.ndarray


def build_grid(min_frequency_hz: float, max_frequency_hz: float, n_frequencies: int, window_length_s: float, n_times: int) -> TimeFrequencyGrid:
    """Build the common time-frequency grid.

    The centre frequencies are ``numpy.logspace`` values between the two limits,
    as in ``obspy.signal.tf_misfit.cwt``.

    Args:
        min_frequency_hz: Lowest centre frequency.
        max_frequency_hz: Highest centre frequency.
        n_frequencies: Number of frequencies.
        window_length_s: Length of the analysis window.
        n_times: Number of time bins.

    Returns:
        TimeFrequencyGrid.
    """
    frequencies = np.logspace(np.log10(min_frequency_hz), np.log10(max_frequency_hz), n_frequencies)
    time_step = window_length_s / n_times
    return grid_from_axes(frequencies, (np.arange(n_times) + 0.5) * time_step)


def grid_from_axes(frequencies_hz: np.ndarray, times_s: np.ndarray) -> TimeFrequencyGrid:
    """Build a grid from logarithmically spaced centre frequencies and time-bin centres.

    Band edges are the geometric midpoints between centres, extended by half a
    step beyond the first and last centre.

    Args:
        frequencies_hz: Centre frequencies (logarithmically spaced, increasing).
        times_s: Time-bin centres.

    Returns:
        TimeFrequencyGrid.
    """
    log_centres = np.log10(frequencies_hz)
    step = log_centres[1] - log_centres[0]
    log_edges = np.concatenate(
        ([log_centres[0] - step / 2], (log_centres[:-1] + log_centres[1:]) / 2, [log_centres[-1] + step / 2])
    )
    return TimeFrequencyGrid(frequencies_hz=np.asarray(frequencies_hz), frequency_edges_hz=10.0**log_edges, times_s=np.asarray(times_s))


def stft_power(
    segment: np.ndarray, sampling_rate_hz: float, frame_centres_s: np.ndarray, window_s: float, window_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectral density of frames centred at given times.

    Args:
        segment: Array of shape (components, samples).
        sampling_rate_hz: Sampling rate of the segment.
        frame_centres_s: Frame centres in seconds from the segment start.
        window_s: Frame length in seconds.
        window_type: Window name accepted by ``scipy.signal.get_window``.

    Returns:
        Tuple (linear frequencies, power of shape (components, frequencies, frames)).

    Raises:
        ValueError: If a frame extends beyond the segment.
    """
    frame_length = int(round(window_s * sampling_rate_hz))
    half = frame_length // 2
    centres = np.round(frame_centres_s * sampling_rate_hz).astype(int)
    starts = centres - half
    if starts.min() < 0 or starts.max() + frame_length > segment.shape[-1]:
        raise ValueError("STFT frames extend beyond the segment: increase the context margin.")
    taper = get_window(window_type, frame_length, fftbins=True)
    frame_index = starts[:, None] + np.arange(frame_length)[None, :]
    frames = segment[:, frame_index] * taper
    spectrum = np.fft.rfft(frames, axis=-1)
    power = np.abs(spectrum) ** 2 / (sampling_rate_hz * np.sum(taper**2))
    power[..., 1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / sampling_rate_hz)
    return frequencies, np.moveaxis(power, -1, -2)


def bin_to_log_bands(linear_frequencies: np.ndarray, power: np.ndarray, grid: TimeFrequencyGrid) -> np.ndarray:
    """Average power within logarithmic bands; interpolate bands without linear bins.

    Args:
        linear_frequencies: Frequencies of the linear bins.
        power: Array of shape (components, linear frequencies, times).
        grid: Target grid.

    Returns:
        Array of shape (components, grid frequencies, times).
    """
    edges = grid.frequency_edges_hz
    output = np.empty((power.shape[0], len(grid.frequencies_hz), power.shape[-1]))
    for band, centre in enumerate(grid.frequencies_hz):
        inside = (linear_frequencies >= edges[band]) & (linear_frequencies < edges[band + 1])
        if inside.any():
            output[:, band, :] = power[:, inside, :].mean(axis=1)
        else:
            upper = np.searchsorted(linear_frequencies, centre)
            lower = upper - 1
            weight = (centre - linear_frequencies[lower]) / (linear_frequencies[upper] - linear_frequencies[lower])
            output[:, band, :] = (1.0 - weight) * power[:, lower, :] + weight * power[:, upper, :]
    return output


def spectrogram(
    segment: np.ndarray,
    sampling_rate_hz: float,
    grid: TimeFrequencyGrid,
    margin_s: float,
    window_s: float,
    window_type: str,
) -> np.ndarray:
    """Spectrogram power on the common grid.

    Args:
        segment: Window extended by the margin, shape (components, samples).
        sampling_rate_hz: Sampling rate of the segment.
        grid: Common grid.
        margin_s: Context margin at the start of the segment.
        window_s: STFT frame length.
        window_type: STFT window name.

    Returns:
        Array of shape (components, frequencies, times).
    """
    frequencies, power = stft_power(segment, sampling_rate_hz, margin_s + grid.times_s, window_s, window_type)
    return bin_to_log_bands(frequencies, power, grid)


def average_time_bins(power: np.ndarray, sampling_rate_hz: float, margin_s: float, window_length_s: float, n_times: int) -> np.ndarray:
    """Average sample-wise power within the time bins of the window.

    Args:
        power: Array of shape (components, frequencies, samples) covering the segment.
        sampling_rate_hz: Sampling rate.
        margin_s: Context margin at the start of the segment.
        window_length_s: Length of the analysis window.
        n_times: Number of time bins.

    Returns:
        Array of shape (components, frequencies, n_times).
    """
    edges = np.round((margin_s + np.linspace(0.0, window_length_s, n_times + 1)) * sampling_rate_hz).astype(int)
    sums = np.add.reduceat(power[..., : edges[-1]], edges[:-1], axis=-1)
    return sums / np.diff(edges)


def morlet_power(segment: np.ndarray, sampling_rate_hz: float, frequencies_hz: np.ndarray, morlet_w0: float) -> np.ndarray:
    """Sample-wise power of the Morlet continuous wavelet transform.

    For the scale ``a = w0 / (2 pi f)`` the transform is
    ``W(a, b) = integral x(t) a^(-1/2) psi*((t - b) / a) dt`` with
    ``psi(t) = pi^(-1/4) exp(i w0 t) exp(-t^2 / 2)``, as in
    ``obspy.signal.tf_misfit.cwt``. It is computed as an inverse FFT of the
    signal spectrum times ``a^(1/2) Psi(a omega)``, where
    ``Psi(omega) = pi^(-1/4) sqrt(2 pi) exp(-(omega - w0)^2 / 2)`` is the Fourier
    transform of the wavelet. The signal is zero-padded to at least twice its
    length to avoid circular wrap-around. Computed in single precision.

    Args:
        segment: Array of shape (components, samples).
        sampling_rate_hz: Sampling rate.
        frequencies_hz: Analysis frequencies.
        morlet_w0: Morlet parameter.

    Returns:
        Array of shape (components, frequencies, samples).
    """
    n_samples = segment.shape[-1]
    n_fft = scipy_fft.next_fast_len(2 * n_samples)
    spectrum = scipy_fft.fft(segment.astype(np.float32), n=n_fft, axis=-1, workers=-1)
    angular_frequencies = (2.0 * np.pi * scipy_fft.fftfreq(n_fft, d=1.0 / sampling_rate_hz)).astype(np.float32)
    scales = (morlet_w0 / (2.0 * np.pi * np.asarray(frequencies_hz))).astype(np.float32)
    kernel = (
        np.pi**-0.25
        * np.sqrt(2.0 * np.pi)
        * np.sqrt(scales)[:, None]
        * np.exp(-0.5 * (scales[:, None] * angular_frequencies[None, :] - morlet_w0) ** 2)
    ).astype(np.float32)
    coefficients = scipy_fft.ifft(spectrum[:, None, :] * kernel[None], axis=-1, workers=-1)[..., :n_samples]
    return np.abs(coefficients) ** 2


def scalogram(
    segment: np.ndarray,
    sampling_rate_hz: float,
    grid: TimeFrequencyGrid,
    margin_s: float,
    window_length_s: float,
    morlet_w0: float,
) -> np.ndarray:
    """Morlet scalogram power on the common grid.

    Args:
        segment: Window extended by the margin, shape (components, samples).
        sampling_rate_hz: Sampling rate of the segment.
        grid: Common grid.
        margin_s: Context margin at the start of the segment.
        window_length_s: Length of the analysis window.
        morlet_w0: Morlet parameter.

    Returns:
        Array of shape (components, frequencies, times).
    """
    power = morlet_power(segment, sampling_rate_hz, grid.frequencies_hz, morlet_w0)
    return average_time_bins(power, sampling_rate_hz, margin_s, window_length_s, len(grid.times_s))


def power_to_db(power: np.ndarray, reference_power: float, floor_power: float) -> np.ndarray:
    """Convert power to decibels relative to a fixed reference.

    Args:
        power: Power values.
        reference_power: Reference of the decibel scale (0 dB).
        floor_power: Values below this are raised to it before the logarithm.

    Returns:
        Array in decibels.
    """
    return 10.0 * np.log10(np.maximum(power, floor_power) / reference_power)
