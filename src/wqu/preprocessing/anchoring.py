"""Anchors of the analysis window.

Each method returns the sample on which the window is placed, or ``None``
when it cannot be determined for the record:

- ``pga``: largest absolute value over the three components;
- ``envelope``: maximum of the energy envelope (sum over the components of the
  squared amplitudes, averaged over a moving window), insensitive to isolated spikes;
- ``arias``: time at which the cumulative energy (proportional to the Arias
  intensity summed over the components) reaches a given fraction of its total;
- ``theoretical``: arrival time of a phase computed from the catalogue origin time,
  the hypocentral distance and a constant velocity;
- ``hybrid``: maximum of the energy envelope within an interval around the
  theoretical arrival.

``window_anchor`` gives the anchor used by the pipeline (method set by
``preprocessing.window.anchor``); the other functions compare the methods.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from wqu.config import AnchoringConfig
from wqu.data.loading import ORIGIN_TIME_COLUMN, TRACE_START_COLUMN
from wqu.preprocessing.windowing import PreparedTrace, cumulative_energy, interval_sums, prepare_trace

ANCHOR_METHODS: tuple[str, ...] = ("pga", "envelope", "arias", "theoretical", "hybrid")


def energy_envelope(traces: np.ndarray, sampling_rate_hz: float, window_s: float) -> np.ndarray:
    """Moving average of the squared amplitudes summed over the components.

    Args:
        traces: Mean-removed array of shape (components, samples).
        sampling_rate_hz: Sampling rate.
        window_s: Length of the centred moving window.

    Returns:
        Array of shape (samples,).
    """
    cumulative = cumulative_energy(traces)
    half = max(int(round(window_s * sampling_rate_hz / 2.0)), 1)
    samples = np.arange(traces.shape[-1])
    start, stop = samples - half, samples + half + 1
    return interval_sums(cumulative, start, stop) / (2 * half + 1)


def envelope_peak_sample(envelope: np.ndarray, start: int = 0, stop: int | None = None) -> int | None:
    """Sample of the envelope maximum within ``[start, stop)``.

    Args:
        envelope: Energy envelope.
        start: First sample of the search interval.
        stop: Sample after the last one (end of the envelope when None).

    Returns:
        Sample index, or None when the interval is empty or contains no energy.
    """
    stop = len(envelope) if stop is None else stop
    start, stop = max(start, 0), min(stop, len(envelope))
    if stop <= start or not np.any(envelope[start:stop] > 0):
        return None
    return start + int(np.argmax(envelope[start:stop]))


def arias_sample(traces: np.ndarray, fraction: float) -> int | None:
    """First sample at which the cumulative energy reaches a fraction of its total.

    Args:
        traces: Mean-removed array of shape (components, samples).
        fraction: Fraction of the total energy, in (0, 1).

    Returns:
        Sample index, or None when the record has no energy.
    """
    cumulative = cumulative_energy(traces)[1:]
    total = cumulative[-1]
    if total <= 0:
        return None
    return int(np.searchsorted(cumulative, fraction * total))


def theoretical_arrival_sample(
    record: pd.Series, distance_column: str, velocity_km_s: float, sampling_rate_hz: float, n_samples: int
) -> int | None:
    """Sample of the theoretical arrival of a phase travelling at constant velocity.

    Args:
        record: Metadata row with parsed origin and trace start times and the distance.
        distance_column: Column with the source-station distance in km.
        velocity_km_s: Propagation velocity.
        sampling_rate_hz: Sampling rate of the stored trace.
        n_samples: Length of the stored trace.

    Returns:
        Sample index, or None when a value is missing or the arrival falls outside the trace.
    """
    origin, start, distance = record[ORIGIN_TIME_COLUMN], record[TRACE_START_COLUMN], record[distance_column]
    if pd.isna(origin) or pd.isna(start) or pd.isna(distance):
        return None
    arrival_s = (origin - start).total_seconds() + float(distance) / velocity_km_s
    sample = int(round(arrival_s * sampling_rate_hz))
    return sample if 0 <= sample < n_samples else None


def anchor_sample(
    prepared: PreparedTrace, record: pd.Series, sampling_rate_hz: float, method: str, anchoring: AnchoringConfig
) -> int | None:
    """Anchor sample of one record with one method.

    Args:
        prepared: Prepared trace (mean removed, padding and PGA located).
        record: Metadata row of the record (used by ``theoretical`` and ``hybrid``).
        sampling_rate_hz: Sampling rate of the stored trace.
        method: One of ``ANCHOR_METHODS``.
        anchoring: Anchoring configuration.

    Returns:
        Sample index, or None when the method cannot determine the anchor.

    Raises:
        ValueError: If the method is unknown.
    """
    traces = prepared.centred
    if method == "pga":
        return prepared.pga_index if np.any(traces != 0) else None
    if method == "envelope":
        return envelope_peak_sample(energy_envelope(traces, sampling_rate_hz, anchoring.envelope_window_s))
    if method == "arias":
        return arias_sample(traces, anchoring.arias_fraction)
    if method not in ("theoretical", "hybrid"):
        raise ValueError(f"Unknown anchoring method '{method}'; expected one of {ANCHOR_METHODS}.")
    theoretical = theoretical_arrival_sample(
        record, anchoring.distance_column, anchoring.phase_velocity_km_s, sampling_rate_hz, traces.shape[-1]
    )
    if method == "theoretical" or theoretical is None:
        return theoretical
    half_width = int(round(anchoring.hybrid_search_half_width_s * sampling_rate_hz))
    envelope = energy_envelope(traces, sampling_rate_hz, anchoring.envelope_window_s)
    return envelope_peak_sample(envelope, theoretical - half_width, theoretical + half_width + 1)


def window_anchor(
    prepared: PreparedTrace, record: pd.Series, sampling_rate_hz: float, method: str, anchoring: AnchoringConfig
) -> int:
    """Anchor used to place the analysis window, falling back to the PGA sample.

    When the chosen method cannot determine an anchor (for example a record
    without data, or a missing theoretical arrival) the PGA sample is used, which
    is defined for every record.

    Args:
        prepared: Prepared trace (mean removed, padding and PGA located).
        record: Metadata row of the record.
        sampling_rate_hz: Sampling rate of the stored trace.
        method: One of ``ANCHOR_METHODS``.
        anchoring: Anchoring configuration.

    Returns:
        Sample index.
    """
    sample = anchor_sample(prepared, record, sampling_rate_hz, method, anchoring)
    return prepared.pga_index if sample is None else sample


def record_anchors(
    prepared: PreparedTrace, record: pd.Series, sampling_rate_hz: float, anchoring: AnchoringConfig
) -> dict[str, int | None]:
    """Anchor sample of one record for every method in ``anchoring.methods``.

    Args:
        prepared: Prepared trace (mean removed, padding and PGA located).
        record: Metadata row of the record.
        sampling_rate_hz: Sampling rate of the stored trace.
        anchoring: Anchoring configuration.

    Returns:
        Mapping from method to anchor sample (None when undetermined).
    """
    return {
        method: anchor_sample(prepared, record, sampling_rate_hz, method, anchoring) for method in anchoring.methods
    }


def anchor_table(
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    anchoring: AnchoringConfig,
    pre_anchor_s: float,
    post_anchor_s: float,
    tolerance: float,
) -> pd.DataFrame:
    """Anchor times of every method and energy kept by the window placed on each.

    Args:
        batches: Iterable of (records, traces); records carry the parsed metadata.
        sampling_rate_hz: Sampling rate of the stored traces.
        anchoring: Anchoring configuration.
        pre_anchor_s: Window length before the anchor.
        post_anchor_s: Window length after the anchor.
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        Table indexed like the records with ``anchor_s_<method>`` (seconds from the
        trace start, NaN when undetermined) and ``energy_fraction_kept_<method>``.
    """
    pre = int(round(pre_anchor_s * sampling_rate_hz))
    post = int(round(post_anchor_s * sampling_rate_hz))
    rows = {}
    for records, traces in batches:
        for (index, record), trace in zip(records.iterrows(), traces):
            prepared = prepare_trace(trace, tolerance)
            cumulative = cumulative_energy(prepared.centred)
            total = cumulative[-1]
            row: dict[str, float] = {}
            for method, sample in record_anchors(prepared, record, sampling_rate_hz, anchoring).items():
                if sample is None or total <= 0:
                    row[f"anchor_s_{method}"] = np.nan if sample is None else sample / sampling_rate_hz
                    row[f"energy_fraction_kept_{method}"] = np.nan
                    continue
                kept = interval_sums(cumulative, np.array([sample - pre]), np.array([sample + post]))[0]
                row[f"anchor_s_{method}"] = sample / sampling_rate_hz
                row[f"energy_fraction_kept_{method}"] = float(kept / total)
            rows[index] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def missing_anchor_share(table: pd.DataFrame, labels: pd.Series, methods: Sequence[str]) -> pd.DataFrame:
    """Share of records without an anchor, per method and label.

    Args:
        table: Output of ``anchor_table``.
        labels: Labels aligned with ``table``.
        methods: Methods to report.

    Returns:
        Table with one row per method and one column per label.
    """
    missing = pd.DataFrame({method: table[f"anchor_s_{method}"].isna() for method in methods})
    return missing.groupby(labels).mean().T


def anchor_disagreement(
    table: pd.DataFrame, labels: pd.Series, methods: Sequence[str], tolerance_s: float
) -> pd.DataFrame:
    """Pairwise disagreement between anchoring methods, per label.

    Args:
        table: Output of ``anchor_table``.
        labels: Labels aligned with ``table``.
        methods: Methods to compare.
        tolerance_s: Two anchors disagree when they differ by more than this.

    Returns:
        Table indexed by (method pair, label) with ``share_disagreeing`` and
        ``median_abs_difference_s``, computed on records where both anchors exist.
    """
    rows = []
    for first, second in combinations(methods, 2):
        difference = (table[f"anchor_s_{first}"] - table[f"anchor_s_{second}"]).abs()
        for label, values in difference.groupby(labels):
            available = values.dropna()
            rows.append(
                {
                    "pair": f"{first} - {second}",
                    "label": label,
                    "share_disagreeing": float((available > tolerance_s).mean()) if len(available) else np.nan,
                    "median_abs_difference_s": float(available.median()) if len(available) else np.nan,
                }
            )
    return pd.DataFrame(rows).set_index(["pair", "label"])


def energy_kept_by_method(
    table: pd.DataFrame, labels: pd.Series, methods: Sequence[str], quantiles: Sequence[float]
) -> pd.DataFrame:
    """Quantiles of the energy fraction kept by the window placed on each anchor.

    Args:
        table: Output of ``anchor_table``.
        labels: Labels aligned with ``table``.
        methods: Methods to report.
        quantiles: Quantiles to compute.

    Returns:
        Table indexed by (method, label) with one column per quantile.
    """
    frames = []
    for method in methods:
        values = table[f"energy_fraction_kept_{method}"]
        summary = values.groupby(labels).quantile(list(quantiles)).unstack()
        summary.index = pd.MultiIndex.from_product([[method], summary.index], names=["method", "label"])
        frames.append(summary)
    return pd.concat(frames)
