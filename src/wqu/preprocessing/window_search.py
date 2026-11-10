"""Grid search over the analysis window placed around its anchor.

Each candidate window goes from ``pre`` seconds before to ``post`` seconds after
the anchor (``preprocessing.window.anchor``). Two properties are measured for every candidate:

- leakage of the zero padding: samples of the window that fall in the leading or
  trailing zero padding of a component, or outside the stored trace. The padding
  must carry no information on the class. It is tested in two ways:

  * univariate: seconds of padding before the anchor, after the anchor and in total
    (worst component), each compared between the classes with the ROC AUC;
  * multivariate: a classifier that sees only the padding (seconds before and
    after the anchor per component, and the fraction of padding in time bins of the
    window per component) is evaluated out of fold, with folds grouped by event.
    Its AUC measures whether the padding as a whole, not only its amount,
    differs between the classes.

  Each test gives a separation max(AUC, 1 - AUC), where 0.5 means no information;
  the window is accepted when the largest of them is within the threshold;
- signal kept: fraction of the energy (sum of squared amplitudes over the three
  components) and of the recorded samples of the stored trace inside the window.

The padding of a record depends only on its zero padding at both ends and on the
anchor sample, so the traces are read once: the padding of every window is computed
afterwards from these few numbers. The energy before and after the anchor is also
stored per candidate half, and each combination is obtained by adding the halves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from wqu.config import AnchoringConfig, WindowSearchConfig
from wqu.data.exploration import numeric_univariate_auc
from wqu.preprocessing.anchoring import window_anchor
from wqu.preprocessing.windowing import PreparedTrace, cumulative_energy, interval_sums, prepare_trace

PADDING_MEASURES: tuple[str, ...] = ("padding_before_anchor_s", "padding_after_anchor_s", "padding_total_s")
CLASSIFIER_SEPARATION_COLUMN = "separation_padding_classifier"
PADDING_SEPARATION_COLUMN = "padding_separation"
ENERGY_SCORE_COLUMN = "energy_score"
WINDOW_LENGTH_COLUMN = "window_length_s"


@dataclass(frozen=True)
class PaddingGeometry:
    """Zero padding and anchor of every record, enough to locate the padding of any window.

    Attributes:
        leading: Zero samples at the start of each component, shape (records, components).
        trailing: Zero samples at the end of each component, shape (records, components).
        anchor_index: Anchor sample of each record, shape (records,).
        n_samples: Length of the stored traces.
        sampling_rate_hz: Sampling rate of the stored traces.
    """

    leading: np.ndarray
    trailing: np.ndarray
    anchor_index: np.ndarray
    n_samples: int
    sampling_rate_hz: float

    @property
    def has_empty_component(self) -> np.ndarray:
        """Boolean array, true for records with at least one component without data."""
        return (self.leading >= self.n_samples).any(axis=1)


@dataclass(frozen=True)
class WindowGridMetrics:
    """Measures of all candidate windows.

    Attributes:
        pre_anchor_s: Candidate lengths before the anchor, shape (n_pre,).
        post_anchor_s: Candidate lengths after the anchor, shape (n_post,).
        index: Record index, length n_records.
        labels: Record labels, shape (n_records,).
        groups: Record groups (events) used for the cross-validation folds, shape (n_records,).
        geometry: Zero padding and anchor of the records.
        energy_fraction_kept: Shape (n_records, n_pre, n_post).
        recorded_fraction_kept: Shape (n_records, n_pre, n_post).
    """

    pre_anchor_s: np.ndarray
    post_anchor_s: np.ndarray
    index: pd.Index
    labels: np.ndarray
    groups: np.ndarray
    geometry: PaddingGeometry
    energy_fraction_kept: np.ndarray
    recorded_fraction_kept: np.ndarray


@dataclass(frozen=True)
class RecordEnergyProfile:
    """Energy and recorded samples in the two halves of the candidate windows of one record.

    Attributes:
        energy_before: Energy before the anchor, shape (n_pre,).
        energy_after: Energy from the anchor on, shape (n_post,).
        recorded_before: Recorded samples before the anchor (all components), shape (n_pre,).
        recorded_after: Recorded samples from the anchor on, shape (n_post,).
        total_energy: Energy of the stored trace.
        total_recorded: Recorded samples of the stored trace.
    """

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
        start: Interval starts, shape (records, intervals).
        stop: Interval stops, shape (records, intervals).
        leading: Zero samples at the start of each component, shape (records, components).
        trailing: Zero samples at the end of each component, shape (records, components).
        n_samples: Length of the stored traces.

    Returns:
        Array of shape (records, components, intervals).
    """
    start = np.asarray(start)[:, None, :]
    stop = np.asarray(stop)[:, None, :]
    lead = np.asarray(leading)[:, :, None]
    tail_start = np.maximum(n_samples - np.asarray(trailing)[:, :, None], lead)
    in_leading = np.clip(np.minimum(stop, lead) - start, 0, None)
    in_trailing = np.clip(stop - np.maximum(start, tail_start), 0, None)
    return in_leading + in_trailing


def window_padding(
    geometry: PaddingGeometry, pre_anchor_s: float, post_anchor_s: float, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padding of one window for every record.

    Args:
        geometry: Zero padding and anchor of the records.
        pre_anchor_s: Window length before the anchor.
        post_anchor_s: Window length after the anchor.
        n_bins: Number of equal time bins of the window used for the padding mask.

    Returns:
        Tuple (seconds before the anchor, seconds after the anchor), each of shape
        (records, components), and the padding fraction of each time bin, of shape
        (records, components, n_bins).
    """
    rate = geometry.sampling_rate_hz
    anchor = geometry.anchor_index[:, None]
    pre = int(round(pre_anchor_s * rate))
    post = int(round(post_anchor_s * rate))
    halves_start = np.hstack([anchor - pre, anchor])
    halves_stop = np.hstack([anchor, anchor + post])
    halves = padding_samples(halves_start, halves_stop, geometry.leading, geometry.trailing, geometry.n_samples)
    edges = np.round(np.linspace(-pre, post, n_bins + 1)).astype(int)
    bins = padding_samples(anchor + edges[:-1], anchor + edges[1:], geometry.leading, geometry.trailing, geometry.n_samples)
    return halves[..., 0] / rate, halves[..., 1] / rate, bins / np.diff(edges)


def padding_measures(before_s: np.ndarray, after_s: np.ndarray, index: pd.Index) -> pd.DataFrame:
    """Scalar padding measures of one window (worst component).

    Args:
        before_s: Padding seconds before the anchor, shape (records, components).
        after_s: Padding seconds after the anchor, shape (records, components).
        index: Record index.

    Returns:
        Table with the columns of ``PADDING_MEASURES``.
    """
    return pd.DataFrame(
        {
            "padding_before_anchor_s": before_s.max(axis=1),
            "padding_after_anchor_s": after_s.max(axis=1),
            "padding_total_s": (before_s + after_s).max(axis=1),
        },
        index=index,
    )


def padding_features(before_s: np.ndarray, after_s: np.ndarray, bin_fractions: np.ndarray) -> np.ndarray:
    """Feature matrix of the padding classifier.

    Args:
        before_s: Padding seconds before the anchor, shape (records, components).
        after_s: Padding seconds after the anchor, shape (records, components).
        bin_fractions: Padding fraction per time bin, shape (records, components, bins).

    Returns:
        Array of shape (records, 2 * components + components * bins).
    """
    return np.hstack([before_s, after_s, bin_fractions.reshape(len(bin_fractions), -1)])


def classifier_separation(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> float:
    """Out-of-fold class separation reached by a classifier on the given features.

    Folds are stratified by label and grouped (records of the same event stay in
    the same fold). Gradient-boosted trees are used, so non-linear combinations
    of the features are captured.

    Args:
        features: Feature matrix, shape (records, features).
        labels: Labels (1 = bad).
        groups: Group of each record.
        n_splits: Number of folds.
        seed: Random seed of the folds and of the classifier.

    Returns:
        max(AUC, 1 - AUC) of the out-of-fold scores; 0.5 when the features are constant.
    """
    if np.all(features == features[0]):
        return 0.5
    scores = np.empty(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(features, labels, groups):
        model = HistGradientBoostingClassifier(random_state=seed)
        model.fit(features[train], labels[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    auc = float(roc_auc_score(labels, scores))
    return max(auc, 1.0 - auc)


def record_energy_profile(
    prepared: PreparedTrace, anchor: int, pre_samples: np.ndarray, post_samples: np.ndarray, tolerance: float
) -> RecordEnergyProfile:
    """Energy and recorded samples of the halves of all candidate windows of one record.

    Args:
        prepared: Prepared trace (mean removed, padding located).
        anchor: Anchor sample of the record.
        pre_samples: Candidate lengths before the anchor, in samples.
        post_samples: Candidate lengths after the anchor, in samples.
        tolerance: Zero threshold on the absolute value.

    Returns:
        RecordEnergyProfile.
    """
    trace = prepared.centred
    energy = cumulative_energy(trace)
    recorded = np.concatenate(([0], np.cumsum((np.abs(trace) > tolerance).sum(axis=0))))
    before_start, before_stop = anchor - pre_samples, np.full_like(pre_samples, anchor)
    after_start, after_stop = np.full_like(post_samples, anchor), anchor + post_samples
    return RecordEnergyProfile(
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
    pre_anchor_s: Sequence[float],
    post_anchor_s: Sequence[float],
    tolerance: float,
    label_column: str,
    group_column: str,
    anchor_method: str,
    anchoring: AnchoringConfig,
) -> WindowGridMetrics:
    """Read every record once and measure every candidate window.

    Args:
        batches: Iterable of (records, traces), traces of shape (records, components, samples);
            the records carry the metadata used by the anchoring methods.
        sampling_rate_hz: Sampling rate of the stored traces.
        pre_anchor_s: Candidate lengths before the anchor.
        post_anchor_s: Candidate lengths after the anchor.
        tolerance: Zero threshold on the absolute value.
        label_column: Label column of the records.
        group_column: Column grouping the records for the cross-validation folds (event).
        anchor_method: Anchoring method (see ``wqu.preprocessing.anchoring``).
        anchoring: Parameters of the anchoring methods.

    Returns:
        WindowGridMetrics.
    """
    pre = np.asarray(sorted(set(pre_anchor_s)), dtype=float)
    post = np.asarray(sorted(set(post_anchor_s)), dtype=float)
    pre_samples = np.round(pre * sampling_rate_hz).astype(int)
    post_samples = np.round(post * sampling_rate_hz).astype(int)
    indices, labels, groups, leading, trailing, anchors = [], [], [], [], [], []
    energy_kept, recorded_kept = [], []
    n_samples = 0
    for records, traces in batches:
        n_samples = traces.shape[-1]
        for (index, record), trace in zip(records.iterrows(), traces):
            prepared = prepare_trace(trace, tolerance)
            anchor = window_anchor(prepared, record, sampling_rate_hz, anchor_method, anchoring)
            profile = record_energy_profile(prepared, anchor, pre_samples, post_samples, tolerance)
            indices.append(index)
            labels.append(int(record[label_column]))
            groups.append(record[group_column])
            leading.append(prepared.leading)
            trailing.append(prepared.trailing)
            anchors.append(anchor)
            energy_kept.append(
                safe_fraction(profile.energy_before[:, None] + profile.energy_after[None, :], profile.total_energy)
            )
            recorded_kept.append(
                safe_fraction(profile.recorded_before[:, None] + profile.recorded_after[None, :], profile.total_recorded)
            )
    geometry = PaddingGeometry(
        leading=np.asarray(leading),
        trailing=np.asarray(trailing),
        anchor_index=np.asarray(anchors),
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz,
    )
    return WindowGridMetrics(
        pre_anchor_s=pre,
        post_anchor_s=post,
        index=pd.Index(indices),
        labels=np.asarray(labels),
        groups=np.asarray(groups),
        geometry=geometry,
        energy_fraction_kept=np.asarray(energy_kept, dtype=np.float32),
        recorded_fraction_kept=np.asarray(recorded_kept, dtype=np.float32),
    )


def leakage_records(metrics: WindowGridMetrics, search: WindowSearchConfig) -> np.ndarray:
    """Records used in the leakage tests.

    Records with a component entirely without data are excluded when configured:
    their padding marks a real defect of the record, not an effect of the window.

    Args:
        metrics: Measures of all candidates.
        search: Window search configuration.

    Returns:
        Boolean array over the records.
    """
    if search.exclude_empty_components:
        return ~metrics.geometry.has_empty_component
    return np.ones(len(metrics.labels), dtype=bool)


@dataclass(frozen=True)
class LeakageInputs:
    """Padding of one window on the records used in the leakage tests.

    Attributes:
        measures: Scalar padding measures (columns of ``PADDING_MEASURES``).
        features: Feature matrix of the padding classifier.
        labels: Labels of the records.
        groups: Groups (events) of the records.
    """

    measures: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray


def leakage_inputs(
    metrics: WindowGridMetrics, pre_anchor_s: float, post_anchor_s: float, search: WindowSearchConfig
) -> LeakageInputs:
    """Padding measures and classifier features of one window.

    Args:
        metrics: Measures of all candidates.
        pre_anchor_s: Window length before the anchor.
        post_anchor_s: Window length after the anchor.
        search: Window search configuration.

    Returns:
        LeakageInputs restricted to the records of ``leakage_records``.
    """
    selected = leakage_records(metrics, search)
    geometry = metrics.geometry
    subset = PaddingGeometry(
        leading=geometry.leading[selected],
        trailing=geometry.trailing[selected],
        anchor_index=geometry.anchor_index[selected],
        n_samples=geometry.n_samples,
        sampling_rate_hz=geometry.sampling_rate_hz,
    )
    before, after, bin_fractions = window_padding(subset, pre_anchor_s, post_anchor_s, search.classifier_mask_bins)
    return LeakageInputs(
        measures=padding_measures(before, after, metrics.index[selected]),
        features=padding_features(before, after, bin_fractions),
        labels=metrics.labels[selected],
        groups=metrics.groups[selected],
    )


def padding_leakage(
    metrics: WindowGridMetrics, pre_anchor_s: float, post_anchor_s: float, search: WindowSearchConfig
) -> dict[str, float]:
    """Univariate and multivariate class separation of the padding of one window.

    Args:
        metrics: Measures of all candidates.
        pre_anchor_s: Window length before the anchor.
        post_anchor_s: Window length after the anchor.
        search: Window search configuration.

    Returns:
        Dictionary with ``separation_<measure>`` for each of ``PADDING_MEASURES``,
        ``separation_padding_classifier``, ``padding_separation`` (largest of them),
        ``share_with_padding_<label>`` (records with any padding in the window) and
        ``n_leakage_records``.
    """
    inputs = leakage_inputs(metrics, pre_anchor_s, post_anchor_s, search)
    label_series = pd.Series(inputs.labels, index=inputs.measures.index)
    result: dict[str, float] = {}
    for measure in PADDING_MEASURES:
        separation = numeric_univariate_auc(inputs.measures[measure], label_series)["separation"]
        result[f"separation_{measure}"] = 0.5 if np.isnan(separation) else separation
    result[CLASSIFIER_SEPARATION_COLUMN] = classifier_separation(
        inputs.features, inputs.labels, inputs.groups, search.classifier_n_splits, search.classifier_random_seed
    )
    result[PADDING_SEPARATION_COLUMN] = max(
        result[CLASSIFIER_SEPARATION_COLUMN], *(result[f"separation_{measure}"] for measure in PADDING_MEASURES)
    )
    for label, share in (inputs.measures["padding_total_s"] > 0).groupby(label_series).mean().items():
        result[f"share_with_padding_{label}"] = float(share)
    result["n_leakage_records"] = len(inputs.labels)
    return result


def classifier_null_separations(
    metrics: WindowGridMetrics, pre_anchor_s: float, post_anchor_s: float, search: WindowSearchConfig
) -> np.ndarray:
    """Separation of the padding classifier after random permutations of the labels.

    With permuted labels the padding carries no class information, so these values
    show the separation reached by chance (finite sample and overfitting) and are the
    reference against which the observed separation and the threshold are judged.
    The groups are kept, so the folds are built as in the real test.

    Args:
        metrics: Measures of all candidates.
        pre_anchor_s: Window length before the anchor.
        post_anchor_s: Window length after the anchor.
        search: Window search configuration.

    Returns:
        Array of shape (n_permutations,).
    """
    inputs = leakage_inputs(metrics, pre_anchor_s, post_anchor_s, search)
    generator = np.random.default_rng(search.classifier_random_seed)
    return np.array(
        [
            classifier_separation(
                inputs.features,
                generator.permutation(inputs.labels),
                inputs.groups,
                search.classifier_n_splits,
                search.classifier_random_seed,
            )
            for _ in range(search.classifier_n_permutations)
        ]
    )


def null_comparison(observed: float, null_separations: np.ndarray) -> dict[str, float]:
    """Compare an observed separation with its permutation distribution.

    Args:
        observed: Separation with the true labels.
        null_separations: Separations with permuted labels.

    Returns:
        Dictionary with ``observed``, ``null_mean``, ``null_95th_percentile`` and
        ``p_value`` (share of permutations at least as separating, with the usual +1 correction).
    """
    return {
        "observed": observed,
        "null_mean": float(null_separations.mean()),
        "null_95th_percentile": float(np.quantile(null_separations, 0.95)),
        "p_value": float((1 + np.sum(null_separations >= observed)) / (1 + len(null_separations))),
    }


def energy_score(energy_fraction_kept: np.ndarray, search: WindowSearchConfig) -> float:
    """Signal-loss score of one window on the reference records (higher is better).

    Args:
        energy_fraction_kept: Energy fraction kept by each reference record.
        search: Window search configuration.

    Returns:
        Quantile of the fraction kept (``quantile`` criterion) or share of records
        keeping at least ``min_energy_fraction`` (``share_above`` criterion).
    """
    if search.energy_criterion == "quantile":
        return float(np.quantile(energy_fraction_kept, search.energy_quantile))
    return float(np.mean(energy_fraction_kept >= search.min_energy_fraction))


def summarize_window_grid(metrics: WindowGridMetrics, search: WindowSearchConfig) -> pd.DataFrame:
    """Leakage and signal-loss measures of every candidate window.

    Args:
        metrics: Measures of all candidates.
        search: Window search configuration.

    Returns:
        Table with one row per (pre_anchor_s, post_anchor_s): window length, padding separations
        (univariate and classifier), share of records with padding per label, energy score,
        median energy and recorded fractions kept on the reference records, and
        ``accepted`` (padding separation within the threshold).

    Raises:
        ValueError: If no record has the reference label.
    """
    reference = metrics.labels == search.energy_reference_label
    if not reference.any():
        raise ValueError(f"No record with the reference label {search.energy_reference_label}.")
    rows = []
    for pre_position, pre in enumerate(metrics.pre_anchor_s):
        for post_position, post in enumerate(metrics.post_anchor_s):
            energy = metrics.energy_fraction_kept[reference, pre_position, post_position]
            recorded = metrics.recorded_fraction_kept[reference, pre_position, post_position]
            rows.append(
                {
                    "pre_anchor_s": pre,
                    "post_anchor_s": post,
                    WINDOW_LENGTH_COLUMN: pre + post,
                    **padding_leakage(metrics, pre, post, search),
                    ENERGY_SCORE_COLUMN: energy_score(energy, search),
                    "median_energy_fraction_kept": float(np.median(energy)),
                    "median_recorded_fraction_kept": float(np.median(recorded)),
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


def window_row(summary: pd.DataFrame, pre_anchor_s: float, post_anchor_s: float) -> pd.Series:
    """Row of a given window in the summary.

    Args:
        summary: Output of ``summarize_window_grid``.
        pre_anchor_s: Length before the anchor.
        post_anchor_s: Length after the anchor.

    Returns:
        Row of the window.

    Raises:
        KeyError: If the window is not in the grid.
    """
    match = summary[np.isclose(summary["pre_anchor_s"], pre_anchor_s) & np.isclose(summary["post_anchor_s"], post_anchor_s)]
    if match.empty:
        raise KeyError(f"Window ({pre_anchor_s} s, {post_anchor_s} s) is not in the grid.")
    return match.iloc[0]
