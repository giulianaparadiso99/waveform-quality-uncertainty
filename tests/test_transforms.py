"""Comparison of the time-frequency transforms with reference implementations.

- ``stft_power`` against ``scipy.signal.spectrogram`` (Hann window, one-sided power
  spectral density), with frames placed where SciPy places them;
- ``morlet_power`` against ``obspy.signal.tf_misfit.cwt`` (same Morlet wavelet,
  normalization and frequencies). ObsPy centres its sampled wavelet half a sample
  away from the analysed sample (the centre of its time axis falls between two
  samples), so its output at sample ``m`` is the transform at ``t_m - dt / 2``; the
  comparison therefore delays the signal by half a sample before ``morlet_power``.
  Close to the highest grid frequency the wavelet spectrum extends beyond the Nyquist
  frequency: ObsPy samples the wavelet in time, so that part folds back onto the
  frequencies near Nyquist, while ``morlet_power`` uses the exact (unfolded) spectrum.
  The two then differ slightly where the signal has energy close to Nyquist; the test
  signal is low-pass filtered as in the preprocessing, and the frequencies above
  ``CWT_STRICT_MAX_FREQUENCY_HZ`` are compared with a wider tolerance;
- the two steps that bring the transforms onto the common grid: averaging within
  logarithmic frequency bands (with interpolation where a band has no linear bin)
  and averaging within time bins.

Run with ``pytest tests``. The ObsPy comparison is skipped if ObsPy is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, chirp, sosfiltfilt
from scipy.signal import spectrogram as scipy_spectrogram

from wqu.preprocessing.transform import (
    average_time_bins,
    bin_to_log_bands,
    build_grid,
    morlet_power,
    stft_power,
)

SAMPLING_RATE_HZ = 100.0
DURATION_S = 120.0
MORLET_W0 = 6.0
MIN_FREQUENCY_HZ = 0.1
MAX_FREQUENCY_HZ = 40.0
N_FREQUENCIES = 128
STFT_WINDOW_S = 10.0
STFT_HOP_SAMPLES = 100
LOWPASS_CORNER_HZ = 40.0
LOWPASS_ORDER = 6
# Part of the trace compared with ObsPy: the two implementations pad the signal differently,
# so the edges (where the longest wavelets reach beyond the trace) are left out.
CWT_EDGE_S = 30.0
STFT_RELATIVE_TOLERANCE = 1e-10
# Errors are relative to the largest power of the component; morlet_power works in single precision.
CWT_STRICT_MAX_FREQUENCY_HZ = 35.0
CWT_STRICT_TOLERANCE = 1e-4
# Above CWT_STRICT_MAX_FREQUENCY_HZ: folding of ObsPy's sampled wavelet near Nyquist.
CWT_NEAR_NYQUIST_TOLERANCE = 2e-2
# ObsPy's output at sample m is the transform at t_m - dt / 2.
OBSPY_DELAY_SAMPLES = 0.5
RANDOM_SEED = 0


def synthetic_trace(sampling_rate_hz: float, duration_s: float, seed: int) -> np.ndarray:
    """Three-component test signal: tones, a chirp and noise under a Gaussian envelope.

    The noise is low-pass filtered as in the preprocessing (Butterworth, ``LOWPASS_ORDER``,
    zero phase, corner ``LOWPASS_CORNER_HZ``).

    Args:
        sampling_rate_hz: Sampling rate.
        duration_s: Length of the trace.
        seed: Seed of the noise.

    Returns:
        Array of shape (3, samples).
    """
    times = np.arange(int(round(duration_s * sampling_rate_hz))) / sampling_rate_hz
    envelope = np.exp(-(((times - duration_s / 2) / (duration_s / 4)) ** 2))
    rng = np.random.default_rng(seed)
    signal = (
        np.sin(2 * np.pi * 1.3 * times)
        + 0.5 * chirp(times, 0.2, times[-1], 30.0)
        + 0.4 * np.sin(2 * np.pi * 12.0 * times + 0.3)
        + 0.3 * np.sin(2 * np.pi * 35.0 * times)
    )
    lowpass = butter(LOWPASS_ORDER, LOWPASS_CORNER_HZ, btype="lowpass", fs=sampling_rate_hz, output="sos")
    noise = sosfiltfilt(lowpass, rng.standard_normal(times.size))
    components = [signal * envelope, signal[::-1] * envelope, noise * envelope]
    return np.stack(components)


def fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """Delay a signal by a fraction of a sample (phase shift in the frequency domain, zero-padded).

    Args:
        signal: One-dimensional signal.
        delay_samples: Delay in samples (positive = later).

    Returns:
        Delayed signal of the same length.
    """
    n_fft = 4 * signal.size
    spectrum = np.fft.rfft(signal, n_fft)
    frequencies = np.fft.rfftfreq(n_fft)
    return np.fft.irfft(spectrum * np.exp(-2j * np.pi * frequencies * delay_samples), n_fft)[: signal.size]


def relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    """Largest absolute difference divided by the largest reference value."""
    return float(np.max(np.abs(values - reference)) / np.max(np.abs(reference)))


def relative_error_by_frequency(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Largest absolute difference at each frequency divided by the largest reference value overall.

    Args:
        values: Array of shape (frequencies, samples).
        reference: Array of the same shape.

    Returns:
        Array of shape (frequencies,).
    """
    return np.max(np.abs(values - reference), axis=-1) / np.max(np.abs(reference))


def test_stft_power_matches_scipy_spectrogram() -> None:
    """Same frames, window and density scaling as scipy.signal.spectrogram."""
    trace = synthetic_trace(SAMPLING_RATE_HZ, DURATION_S, RANDOM_SEED)
    frame_length = int(round(STFT_WINDOW_S * SAMPLING_RATE_HZ))
    reference_frequencies, frame_centres_s, reference_power = scipy_spectrogram(
        trace,
        fs=SAMPLING_RATE_HZ,
        window="hann",
        nperseg=frame_length,
        noverlap=frame_length - STFT_HOP_SAMPLES,
        detrend=False,
        scaling="density",
        mode="psd",
    )
    frequencies, power = stft_power(trace, SAMPLING_RATE_HZ, frame_centres_s, STFT_WINDOW_S, "hann")
    np.testing.assert_allclose(frequencies, reference_frequencies)
    assert power.shape == reference_power.shape
    assert relative_error(power, reference_power) < STFT_RELATIVE_TOLERANCE


def test_morlet_power_matches_obspy_cwt() -> None:
    """Same transform as obspy.signal.tf_misfit.cwt once its half-sample offset is accounted for."""
    tf_misfit = pytest.importorskip("obspy.signal.tf_misfit")
    trace = synthetic_trace(SAMPLING_RATE_HZ, DURATION_S, RANDOM_SEED)
    grid = build_grid(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ, N_FREQUENCIES, DURATION_S, 1)
    reference = np.stack(
        [
            np.abs(tf_misfit.cwt(component, 1.0 / SAMPLING_RATE_HZ, MORLET_W0, MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ, N_FREQUENCIES)) ** 2
            for component in trace
        ]
    )
    delayed = np.stack([fractional_delay(component, OBSPY_DELAY_SAMPLES) for component in trace])
    power = morlet_power(delayed, SAMPLING_RATE_HZ, grid.frequencies_hz, MORLET_W0)
    edge = int(round(CWT_EDGE_S * SAMPLING_RATE_HZ))
    inner = slice(edge, trace.shape[-1] - edge)
    assert power.shape == reference.shape
    strict = grid.frequencies_hz <= CWT_STRICT_MAX_FREQUENCY_HZ
    for component in range(trace.shape[0]):
        errors = relative_error_by_frequency(power[component, :, inner], reference[component, :, inner])
        assert errors[strict].max() < CWT_STRICT_TOLERANCE
        assert errors[~strict].max() < CWT_NEAR_NYQUIST_TOLERANCE


def test_band_average_of_linear_spectrum() -> None:
    """Bands with linear bins get the mean of their bins; empty bands get the interpolated value."""
    frame_length = int(round(STFT_WINDOW_S * SAMPLING_RATE_HZ))
    linear_frequencies = np.fft.rfftfreq(frame_length, d=1.0 / SAMPLING_RATE_HZ)
    grid = build_grid(MIN_FREQUENCY_HZ, MAX_FREQUENCY_HZ, N_FREQUENCIES, DURATION_S, 4)
    intercept, slope = 2.0, 0.5
    spectrum = intercept + slope * linear_frequencies
    power = np.broadcast_to(spectrum[None, :, None], (3, linear_frequencies.size, 4))
    bands = bin_to_log_bands(linear_frequencies, power, grid)
    edges = grid.frequency_edges_hz
    for band, centre in enumerate(grid.frequencies_hz):
        inside = (linear_frequencies >= edges[band]) & (linear_frequencies < edges[band + 1])
        expected = spectrum[inside].mean() if inside.any() else intercept + slope * centre
        np.testing.assert_allclose(bands[:, band, :], expected)


def test_average_time_bins_matches_direct_mean() -> None:
    """Each time bin is the mean of the samples it covers, after the context margin."""
    rng = np.random.default_rng(RANDOM_SEED)
    margin_s, window_length_s, n_times = 20.0, 120.0, 128
    n_samples = int(round((window_length_s + 2 * margin_s) * SAMPLING_RATE_HZ))
    power = rng.random((3, 5, n_samples))
    averaged = average_time_bins(power, SAMPLING_RATE_HZ, margin_s, window_length_s, n_times)
    edges = np.round((margin_s + np.linspace(0.0, window_length_s, n_times + 1)) * SAMPLING_RATE_HZ).astype(int)
    expected = np.stack([power[..., start:stop].mean(axis=-1) for start, stop in zip(edges[:-1], edges[1:])], axis=-1)
    np.testing.assert_allclose(averaged, expected)
