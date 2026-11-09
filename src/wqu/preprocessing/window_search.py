"""Grid search over the analysis window anchored on the PGA.

Each candidate window goes from ``pre`` seconds before to ``post`` seconds after
the PGA. Two properties are measured for every record and candidate:

- zero padding inside the window (samples in the leading or trailing zero padding
  of a component, or outside the stored trace), in seconds on the worst component,
  separately before the PGA, after the PGA and in total. If the amount of padding
  differs between the classes, a network can learn the class from the padding
  instead of from the waveform (leakage);
- signal kept: fraction of the energy (sum of squared amplitudes over the three
  components) and of the recorded (non-zero) samples of the stored trace that fall
  inside the window.

The part of the window before the PGA depends only on ``pre`` and the part after
only on ``post``, so each record is processed once and every combination is
obtained by adding the two parts. A window is accepted when the padding does not
separate the classes beyond ``max_padding_separation``; among accepted windows the
one keeping most energy on the reference class is suggested (the shortest one
among equivalent scores).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wqu.config import WindowSearchConfig
from wqu.data.exploration import numeric_univariate_auc
from wqu.preprocessing.windowing import PreparedTrace, prepare_trace

PADDING_MEASURES: tuple[str, ...] = ("padding_before_pga_s", "padding_after_pga_s", "padding_total_s")
PADDING_SEPARATION_COLUMN = "padding_separation"
ENERGY_SCORE_COLUMN = "energy_score"
WINDOW_LENGTH_COLUMN = "window_length_s"


@dataclass(frozen=True)
class WindowGridMetrics:
    """Per-record measures of every candidate window.

    Attributes:
        pre_pga_s: Candidate lengths before the PGA, shape (n_pre,).
        post_pga_s: Candidate lengths after the PGA, shape (n_post,).
        index: Record index, length n_records.
        labels: Record labels, shape (n_records,).
        padding_before_pga_s: Padding before the PGA on the worst component, shape (n_records, n_pre).
        padding_after_pga_s: Padding after the PGA on the worst component, shape (n_records, n_post).
        padding_total_s: Padding in the window on the worst component, shape (n_records, n_pre, n_post).
        energy_fraction_kept: Shape (n_records, n_pre, n_post).
        recorded_fraction_kept: Shape (n_records, n_pre, n_post).
    """

    pre_pga_s: np.ndarray
    post_pga_s: np.ndarray
    index: pd.Index
    labels: np.ndarray
    padding_before_pga_s: np.ndarray
    padding_after_pga_s: np.ndarray
    padding_total_s: np.ndarray
    energy_fraction_kept: np.ndarray
    recorded_fraction_kept: np.ndarray


@dataclass(frozen=True)
class RecordWindowProfile:
    """Measures of the two halves of the candidate windows of one record.

    Attributes:
        padding_before: Padding samples before the PGA per component, shape (components, n_pre).
        padding_after: Padding samples after the PGA per component, shape (components, n_post).
        energy_before: Energy before the PGA, shape (n_pre,).
        energy_after: Energy from the PGA on, shape (n_post,).
        recorded_before: Recorded samples before the PGA (all components), shape (n_pre,).
        recorded_after: Recorded samples from the PGA on, shape (n_post,).
        total_energy: Energy of the stored trace.
        total_recorded: Recorded samples of the stored trace.
    """

    padding_before: np.ndarray
    padding_after: np.ndarray
    energy_before: np.ndarray
    energy_after: np.ndarray
    recorded_before: np.ndarray
    recorded_after: np.ndarray
    total_energy: float
    total_recorded: float


def padding_samples(
    start: np.ndarray, stop: np.ndarray, leading: np.ndarray, trailing: np.ndarray, n_samples: int
) -> np.ndarray:
    """Count samples of intervals ``[start, stop)`` that fall in the zero padding.

    The padding of a component is everything before its first recorded sample
    (including negative indices) and everything from its trailing padding on
    (including indices beyond the stored trace).

    Args:
        start: Interval starts, shape (n,).
        stop: Interval stops, shape (n,).
        leading: Zero samples at the start of each component, shape (components,).
        trailing: Zero samples at the end of each component, shape (components,).
        n_samples: Length of the stored trace.

    Returns:
        Array of shape (components, n).
    """
    start = np.asarray(start)[None, :]
    stop = np.asarray(stop)[None, :]
    lead = np.asarray(leading)[:, None]
    tail_start = np.maximum(n_samples - np.asarray(trailing)[:, None], lead)
    in_leading = np.clip(np.minimum(stop, lead) - start, 0, None)
    in_trailing = np.clip(stop - np.maximum(start, tail_start), 0, None)
    return in_leading + in_trailing


def interval_sums(cumulative: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Sums over intervals ``[start, stop)`` from a cumulative sum starting at zero.

    Args:
        cumulative: Cumulative sum with a leading zero, shape (n_samples + 1,).
        start: Interval starts (clipped to the trace).
        stop: Interval stops (clipped to the trace).

    Returns:
        Array of interval sums.
    """
    last = len(cumulative) - 1
    return cumulative[np.clip(stop, 0, last)] - cumulative[np.clip(start, 0, last)]


def record_window_profile(
    prepared: PreparedTrace, pre_samples: np.ndarray, post_samples: np.ndarray, tolerance: float
) -> RecordWindowProfile:
    """Measure the halves of all candidate windows of one record.

    Args:
        prepared: Prepared trace (mean removed, padding and PGA located).
        pre_samples: Candidate lengths before the PGA, in samples.
        post_samples: Candidate lengths after the PGA, in samples.
        tolerance: Zero threshold on the absolute value.

    Returns:
        RecordWindowProfile.
    """
    trace = prepared.centred
    n_samples = trace.shape[-1]
    pga = prepared.pga_index
    energy = np.concatenate(([0.0], np.cumsum((trace**2).sum(axis=0))))
    recorded = np.concatenate(([0], np.cumsum((np.abs(trace) > tolerance).sum(axis=0))))
    before_start, before_stop = pga - pre_samples, np.full_like(pre_samples, pga)
    after_start, after_stop = np.full_like(post_samples, pga), pga + post_samples
    return RecordWindowProfile(
        padding_before=padding_samples(before_start, before_stop, prepared.leading, prepared.trailing, n_samples),
        padding_after=padding_samples(after_start, after_stop, prepared.leading, prepared.trailing, n_samples),
        energy_before=interval_sums(energy, before_start, before_stop),
        energy_after=interval_sums(energy, after_start, after_stop),
        recorded_before=interval_sums(recorded, before_start, before_stop),
        recorded_after=interval_sums(recorded, after_start, after_stop),
        total_energy=float(energy[-1]),
        total_recorded=float(recorded[-1]),
    )


def safe_fraction(part: np.ndarray, total: float) -> np.ndarray:
    """Divide by a total, returning zeros when the total is zero.

    Args:
        part: Numerators.
        total: Denominator.

    Returns:
        Array of fractions.
    """
    return part / total if total > 0 else np.zeros_like(part, dtype=float)


def collect_window_metrics(
    batches: Iterable[tuple[pd.DataFrame, np.ndarray]],
    sampling_rate_hz: float,
    pre_pga_s: Sequence[float],
    post_pga_s: Sequence[float],
    tolerance: float,
    label_column: str,
) -> WindowGridMetrics:
    """Measure every candidate window on every record.

    Args:
        batches: Iterable of (records, traces), traces of shape (records, components, samples).
        sampling_rate_hz: Sampling rate of the stored traces.
        pre_pga_s: Candidate lengths before the PGA.
        post_pga_s: Candidate lengths after the PGA.
        tolerance: Zero threshold on the absolute value.
        label_column: Label column of the records.

    Returns:
        WindowGridMetrics.
    """
    pre = np.asarray(sorted(set(pre_pga_s)), dtype=float)
    post = np.asarray(sorted(set(post_pga_s)), dtype=float)
    pre_samples = np.round(pre * sampling_rate_hz).astype(int)
    post_samples = np.round(post * sampling_rate_hz).astype(int)
    indices, labels = [], []
    padding_before, padding_after, padding_total, energy_kept, recorded_kept = [], [], [], [], []
    for records, traces in batches:
        for index, label, trace in zip(records.index, records[label_column], traces):
            profile = record_window_profile(prepare_trace(trace, tolerance), pre_samples, post_samples, tolerance)
            indices.append(index)
            labels.append(int(label))
            padding_before.append(profile.padding_before.max(axis=0))
            padding_after.append(profile.padding_after.max(axis=0))
            padding_total.append((profile.padding_before[:, :, None] + profile.padding_after[:, None, :]).max(axis=0))
            energy_kept.append(
                safe_fraction(profile.energy_before[:, None] + profile.energy_after[None, :], profile.total_energy)
            )
            recorded_kept.append(
                safe_fraction(profile.recorded_before[:, None] + profile.recorded_after[None, :], profile.total_recorded)
            )
    return WindowGridMetrics(
        pre_pga_s=pre,
        post_pga_s=post,
        index=pd.Index(indices),
        labels=np.asarray(labels),
        padding_before_pga_s=np.asarray(padding_before, dtype=np.float32) / sampling_rate_hz,
        padding_after_pga_s=np.asarray(padding_after, dtype=np.float32) / sampling_rate_hz,
        padding_total_s=np.asarray(padding_total, dtype=np.float32) / sampling_rate_hz,
        energy_fraction_kept=np.asarray(energy_kept, dtype=np.float32),
        recorded_fraction_kept=np.asarray(recorded_kept, dtype=np.float32),
    )


def window_measures(metrics: WindowGridMetrics, pre_position: int, post_position: int) -> pd.DataFrame:
    """Per-record measures of one candidate window.

    Args:
        metrics: Measures of all candidates.
        pre_position: Position of the window in ``metrics.pre_pga_s``.
        post_position: Position of the window in ``metrics.post_pga_s``.

    Returns:
        Table indexed like the records with the padding measures and the fractions kept.
    """
    return pd.DataFrame(
        {
            "padding_before_pga_s": metrics.padding_before_pga_s[:, pre_position],
            "padding_after_pga_s": metrics.padding_after_pga_s[:, post_position],
            "padding_total_s": metrics.padding_total_s[:, pre_position, post_position],
            "energy_fraction_kept": metrics.energy_fraction_kept[:, pre_position, post_position],
            "recorded_fraction_kept": metrics.recorded_fraction_kept[:, pre_position, post_position],
        },
        index=metrics.index,
    )


def padding_leakage(measures: pd.DataFrame, labels: pd.Series) -> dict[str, float]:
    """Class separation of the padding measures of one window.

    Args:
        measures: Table with the columns of ``PADDING_MEASURES``.
        labels: Labels aligned with ``measures`` (1 = bad).

    Returns:
        Dictionary with ``separation_<measure>`` for each measure, ``padding_separation``
        (their maximum) and ``share_with_padding_<label>`` (records with any padding in the window).
    """
    result: dict[str, float] = {}
    for measure in PADDING_MEASURES:
        separation = numeric_univariate_auc(measures[measure], labels)["separation"]
        result[f"separation_{measure}"] = 0.5 if np.isnan(separation) else separation
    result[PADDING_SEPARATION_COLUMN] = max(result[f"separation_{measure}"] for measure in PADDING_MEASURES)
    has_padding = measures["padding_total_s"] > 0
    for label, share in has_padding.groupby(labels).mean().items():
        result[f"share_with_padding_{label}"] = float(share)
    return result


def energy_score(energy_fraction_kept: pd.Series, search: WindowSearchConfig) -> float:
    """Signal-loss score of one window on the reference records (higher is better).

    Args:
        energy_fraction_kept: Energy fraction kept by each reference record.
        search: Window search configuration.

    Returns:
        Quantile of the fraction kept (``quantile`` criterion) or share of records
        keeping at least ``min_energy_fraction`` (``share_above`` criterion).
    """
    if search.energy_criterion == "quantile":
        return float(energy_fraction_kept.quantile(search.energy_quantile))
    return float((energy_fraction_kept >= search.min_energy_fraction).mean())


def summarize_window_grid(metrics: WindowGridMetrics, search: WindowSearchConfig) -> pd.DataFrame:
    """Leakage and signal-loss measures of every candidate window.

    Args:
        metrics: Measures of all candidates.
        search: Window search configuration (reference label and energy criterion).

    Returns:
        Table with one row per (pre_pga_s, post_pga_s): window length, padding separations,
        share of records with padding per label, energy score, median energy and recorded
        fractions kept on the reference records, and ``accepted`` (padding separation within
        the threshold).
    """
    labels = pd.Series(metrics.labels, index=metrics.index)
    reference = labels == search.energy_reference_label
    if not reference.any():
        raise ValueError(f"No record with the reference label {search.energy_reference_label}.")
    rows = []
    for pre_position, pre in enumerate(metrics.pre_pga_s):
        for post_position, post in enumerate(metrics.post_pga_s):
            measures = window_measures(metrics, pre_position, post_position)
            rows.append(
                {
                    "pre_pga_s": pre,
                    "post_pga_s": post,
                    WINDOW_LENGTH_COLUMN: pre + post,
                    **padding_leakage(measures, labels),
                    ENERGY_SCORE_COLUMN: energy_score(measures.loc[reference, "energy_fraction_kept"], search),
                    "median_energy_fraction_kept": float(measures.loc[reference, "energy_fraction_kept"].median()),
                    "median_recorded_fraction_kept": float(measures.loc[reference, "recorded_fraction_kept"].median()),
                }
            )
    summary = pd.DataFrame(rows)
    summary["accepted"] = summary[PADDING_SEPARATION_COLUMN] <= search.max_padding_separation
    return summary


def select_window(summary: pd.DataFrame, search: WindowSearchConfig) -> pd.Series:
    """Choose the window to suggest.

    Among accepted windows, those whose energy score is within ``score_tolerance``
    of the best are equivalent; the shortest of them is chosen, then the one with
    the lowest padding separation.

    Args:
        summary: Output of ``summarize_window_grid``.
        search: Window search configuration.

    Returns:
        Row of the chosen window.

    Raises:
        ValueError: If no window satisfies the padding threshold.
    """
    accepted = summary[summary["accepted"]]
    if accepted.empty:
        lowest = summary[PADDING_SEPARATION_COLUMN].min()
        raise ValueError(
            f"No window has padding separation <= {search.max_padding_separation}; "
            f"the lowest in the grid is {lowest:.3f}. Extend the grid or relax the threshold."
        )
    best = accepted[ENERGY_SCORE_COLUMN].max()
    equivalent = accepted[accepted[ENERGY_SCORE_COLUMN] >= best - search.score_tolerance]
    return equivalent.sort_values([WINDOW_LENGTH_COLUMN, PADDING_SEPARATION_COLUMN]).iloc[0]


def pareto_front(summary: pd.DataFrame) -> pd.Series:
    """Mark the windows not dominated in (lower padding separation, higher energy score).

    Args:
        summary: Output of ``summarize_window_grid``.

    Returns:
        Boolean Series aligned with ``summary``.
    """
    separation = summary[PADDING_SEPARATION_COLUMN].to_numpy()
    score = summary[ENERGY_SCORE_COLUMN].to_numpy()
    no_worse = (separation[None, :] <= separation[:, None]) & (score[None, :] >= score[:, None])
    strictly_better = (separation[None, :] < separation[:, None]) | (score[None, :] > score[:, None])
    dominated = (no_worse & strictly_better).any(axis=1)
    return pd.Series(~dominated, index=summary.index, name="pareto")


def window_row(summary: pd.DataFrame, pre_pga_s: float, post_pga_s: float) -> pd.Series:
    """Row of a given window in the summary.

    Args:
        summary: Output of ``summarize_window_grid``.
        pre_pga_s: Length before the PGA.
        post_pga_s: Length after the PGA.

    Returns:
        Row of the window.

    Raises:
        KeyError: If the window is not in the grid.
    """
    match = summary[np.isclose(summary["pre_pga_s"], pre_pga_s) & np.isclose(summary["post_pga_s"], post_pga_s)]
    if match.empty:
        raise KeyError(f"Window ({pre_pga_s} s, {post_pga_s} s) is not in the grid.")
    return match.iloc[0]
