"""Grad-CAM analysis by prediction category.

Every validation record falls in one category, defined by
- the MC dropout prediction: right or wrong (probability thresholded at 0.5);
- the uncertainty: low (below a low quantile of the chosen measure), high (above a
  high quantile) or middle;
- the true class: good or bad.

The Grad-CAM map of the predicted class (MC dropout mean, see ``wqu.models.explain``)
is computed for every record and reduced to a few quantities: its profile over time
and over frequency, the share of the map on zero padding, the share before the
anchor, and its instability across MC passes. Maps are normalized to unit sum, so
that the shares are fractions of the attention of the network. The mean map of
every category is accumulated as well, without storing all the maps.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from wqu.evaluation.uncertainty_assessment import MEASURE_PREDICTIONS, predicted_labels
from wqu.models.explain import grad_cam_by_class
from wqu.preprocessing.windowing import prepare_trace

CORRECTNESS_COLUMN = "correctness"
UNCERTAINTY_LEVEL_COLUMN = "uncertainty_level"
TRUE_CLASS_COLUMN = "true_class"
CATEGORY_COLUMN = "category"
CORRECTNESS_LEVELS: tuple[str, ...] = ("wrong", "right")
UNCERTAINTY_LEVELS: tuple[str, ...] = ("low", "middle", "high")
EXTREME_LEVELS: tuple[str, ...] = ("low", "high")
CLASS_NAMES: tuple[str, ...] = ("good", "bad")
MAP_EPSILON = 1e-12


@dataclass(frozen=True)
class GradCamAggregate:
    """Reduced Grad-CAM maps of the predicted class over a set of records.

    Attributes:
        time_profiles: Normalized map summed over frequencies, shape (records, times).
        frequency_profiles: Normalized map summed over times, shape (records, frequencies).
        instability: Mean MC standard deviation over mean map value, per record.
        category_mean_maps: Mean normalized map of each category, shape (frequencies, times).
        category_counts: Records per category.
    """

    time_profiles: np.ndarray
    frequency_profiles: np.ndarray
    instability: np.ndarray
    category_mean_maps: dict[str, np.ndarray]
    category_counts: dict[str, int]


def category_name(correctness: str, uncertainty_level: str, true_class: str) -> str:
    """Name of a category, e.g. ``wrong-low-good``.

    Args:
        correctness: ``right`` or ``wrong``.
        uncertainty_level: ``low``, ``middle`` or ``high``.
        true_class: ``good`` or ``bad``.

    Returns:
        Category name.
    """
    return f"{correctness}-{uncertainty_level}-{true_class}"


def category_order(levels: Iterable[str] = UNCERTAINTY_LEVELS) -> list[str]:
    """Categories in display order: wrong before right, by uncertainty level, good before bad.

    Args:
        levels: Uncertainty levels to include.

    Returns:
        List of category names.
    """
    return [
        category_name(correctness, level, true_class)
        for correctness in CORRECTNESS_LEVELS
        for level in levels
        for true_class in CLASS_NAMES
    ]


def assign_categories(
    predictions: pd.DataFrame, label_column: str, measure: str, low_quantile: float, high_quantile: float
) -> pd.DataFrame:
    """Add correctness, uncertainty level, true class and category to the predictions.

    Args:
        predictions: Validation predictions (``validation_predictions.csv`` of a run).
        label_column: Name of the label column (1 = bad).
        measure: Uncertainty measure (a key of ``MEASURE_PREDICTIONS``).
        low_quantile: Quantile below which the uncertainty is low.
        high_quantile: Quantile above which the uncertainty is high.

    Returns:
        Copy of ``predictions`` with the four new columns.
    """
    table = predictions.copy()
    labels = table[label_column].to_numpy().astype(int)
    predicted = predicted_labels(table[MEASURE_PREDICTIONS[measure]].to_numpy())
    uncertainty = table[measure]
    low, high = uncertainty.quantile(low_quantile), uncertainty.quantile(high_quantile)
    table[CORRECTNESS_COLUMN] = np.where(predicted == labels, "right", "wrong")
    table[UNCERTAINTY_LEVEL_COLUMN] = np.select([uncertainty <= low, uncertainty >= high], ["low", "high"], "middle")
    table[TRUE_CLASS_COLUMN] = np.array(CLASS_NAMES)[labels]
    table[CATEGORY_COLUMN] = [
        category_name(*values)
        for values in table[[CORRECTNESS_COLUMN, UNCERTAINTY_LEVEL_COLUMN, TRUE_CLASS_COLUMN]].itertuples(index=False)
    ]
    return table


def category_counts(categories: pd.DataFrame) -> pd.DataFrame:
    """Records per category, as a table of uncertainty level by correctness and true class.

    Args:
        categories: Output of ``assign_categories``.

    Returns:
        Table with uncertainty levels as rows and (correctness, true class) as columns.
    """
    counts = categories.groupby([UNCERTAINTY_LEVEL_COLUMN, CORRECTNESS_COLUMN, TRUE_CLASS_COLUMN]).size()
    table = counts.unstack([CORRECTNESS_COLUMN, TRUE_CLASS_COLUMN], fill_value=0)
    columns = pd.MultiIndex.from_product([CORRECTNESS_LEVELS, CLASS_NAMES])
    return table.reindex(index=list(UNCERTAINTY_LEVELS), columns=columns, fill_value=0)


def sample_category_records(categories: pd.DataFrame, records_per_category: int, seed: int) -> pd.DataFrame:
    """Draw records at random from each low- and high-uncertainty category.

    Args:
        categories: Output of ``assign_categories``.
        records_per_category: Records per category (all of them if fewer).
        seed: Random seed.

    Returns:
        Drawn rows, ordered by category (``category_order`` of the extreme levels).
    """
    drawn = []
    for category in category_order(EXTREME_LEVELS):
        members = categories[categories[CATEGORY_COLUMN] == category]
        drawn.append(members.sample(n=min(records_per_category, len(members)), random_state=seed))
    return pd.concat(drawn)


def normalize_maps(maps: torch.Tensor) -> torch.Tensor:
    """Scale each map to unit sum (maps that are zero everywhere stay zero).

    Args:
        maps: Non-negative maps of shape (batch, frequencies, times).

    Returns:
        Normalized maps.
    """
    totals = maps.sum(dim=(-2, -1), keepdim=True)
    return maps / totals.clamp_min(MAP_EPSILON)


def predicted_class_maps(
    model: nn.Module, layer: nn.Module, inputs: torch.Tensor, predicted: np.ndarray, n_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """MC mean and standard deviation of the Grad-CAM map of the predicted class of each record.

    Args:
        model: Trained network.
        layer: Layer explained by Grad-CAM.
        inputs: Batch of standardized images.
        predicted: Predicted label of each record (1 = bad).
        n_samples: MC dropout passes.

    Returns:
        Tuple (mean maps, standard-deviation maps), each of shape (batch, frequencies, times), on the CPU.
    """
    maps = grad_cam_by_class(model, layer, inputs, n_samples)
    is_bad = torch.as_tensor(predicted.astype(bool)).view(-1, 1, 1)
    mean = torch.where(is_bad, maps["bad"].mean.cpu(), maps["good"].mean.cpu())
    std = torch.where(is_bad, maps["bad"].std.cpu(), maps["good"].std.cpu())
    return mean, std


def aggregate_grad_cam(
    model: nn.Module,
    layer: nn.Module,
    loader: DataLoader,
    categories: pd.DataFrame,
    prediction_column: str,
    n_samples: int,
    device: str,
) -> GradCamAggregate:
    """Grad-CAM map of the predicted class of every record, reduced to profiles and category means.

    Args:
        model: Trained network.
        layer: Layer explained by Grad-CAM.
        loader: Loader of the records of ``categories``, in the same order, not shuffled.
        categories: Output of ``assign_categories``.
        prediction_column: Probability column defining the predicted class.
        n_samples: MC dropout passes per map.
        device: Device name.

    Returns:
        GradCamAggregate.
    """
    model.to(device)
    predicted = predicted_labels(categories[prediction_column].to_numpy())
    names = categories[CATEGORY_COLUMN].to_numpy()
    time_profiles, frequency_profiles, instability = [], [], []
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    start = 0
    for inputs, _ in loader:
        stop = start + len(inputs)
        mean, std = predicted_class_maps(model, layer, inputs.to(device), predicted[start:stop], n_samples)
        normalized = normalize_maps(mean)
        time_profiles.append(normalized.sum(dim=-2).numpy())
        frequency_profiles.append(normalized.sum(dim=-1).numpy())
        instability.append((std.mean(dim=(-2, -1)) / mean.mean(dim=(-2, -1)).clamp_min(MAP_EPSILON)).numpy())
        for position, name in enumerate(names[start:stop]):
            sums[name] = sums.get(name, 0.0) + normalized[position].numpy()
            counts[name] = counts.get(name, 0) + 1
        start = stop
    return GradCamAggregate(
        time_profiles=np.concatenate(time_profiles),
        frequency_profiles=np.concatenate(frequency_profiles),
        instability=np.concatenate(instability),
        category_mean_maps={name: total / counts[name] for name, total in sums.items()},
        category_counts=counts,
    )


def time_bin_edges(times_s: np.ndarray) -> np.ndarray:
    """Edges of equally spaced time bins from their centres.

    Args:
        times_s: Bin centres (seconds from the window start).

    Returns:
        Array of ``len(times_s) + 1`` edges.
    """
    step = times_s[1] - times_s[0]
    return np.concatenate((times_s - step / 2, [times_s[-1] + step / 2]))


def padding_profile(
    trace: np.ndarray,
    window_start: int,
    window_stop: int,
    sampling_rate_hz: float,
    times_s: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Share of zero-padded samples (mean over components) in each time bin of the window.

    A sample is padding when it lies outside the trace or in the leading or trailing
    zeros of its component.

    Args:
        trace: Stored trace, shape (components, samples).
        window_start: First sample of the window (may be negative).
        window_stop: Sample after the last one of the window (may exceed the trace).
        sampling_rate_hz: Sampling rate of the stored trace.
        times_s: Time-bin centres of the grid (seconds from the window start).
        tolerance: Zero threshold used to find the zero padding.

    Returns:
        Array of shape (times,) with values in [0, 1].
    """
    prepared = prepare_trace(trace, tolerance)
    n_samples = trace.shape[-1]
    samples = np.arange(window_start, window_stop)
    recorded = np.stack(
        [(samples >= leading) & (samples < n_samples - trailing) for leading, trailing in zip(prepared.leading, prepared.trailing)]
    )
    padded = 1.0 - recorded.mean(axis=0)
    bins = np.clip(np.digitize((samples - window_start) / sampling_rate_hz, time_bin_edges(times_s)) - 1, 0, len(times_s) - 1)
    totals = np.bincount(bins, weights=padded, minlength=len(times_s))
    sizes = np.bincount(bins, minlength=len(times_s))
    return totals / np.maximum(sizes, 1)


def map_summary(
    aggregate: GradCamAggregate, padding_profiles: np.ndarray, times_s: np.ndarray, pre_anchor_s: float
) -> pd.DataFrame:
    """Per-record shares of the map on padding and before the anchor, and map instability.

    Args:
        aggregate: Output of ``aggregate_grad_cam``.
        padding_profiles: Output of ``padding_profile`` for every record, shape (records, times).
        times_s: Time-bin centres of the grid.
        pre_anchor_s: Seconds of the window before the anchor.

    Returns:
        Table with ``padding_share``, ``window_padding_share`` (share of the window that
        is padding, for reference), ``pre_anchor_share`` and ``instability``.
    """
    before_anchor = times_s < pre_anchor_s
    return pd.DataFrame(
        {
            "padding_share": (aggregate.time_profiles * padding_profiles).sum(axis=1),
            "window_padding_share": padding_profiles.mean(axis=1),
            "pre_anchor_share": aggregate.time_profiles[:, before_anchor].sum(axis=1),
            "instability": aggregate.instability,
        }
    )


def summary_by_category(summary: pd.DataFrame, categories: pd.Series) -> pd.DataFrame:
    """Median of each map quantity per category, with the number of records.

    Args:
        summary: Output of ``map_summary``.
        categories: Category of each record, aligned with ``summary``.

    Returns:
        Table with one row per category, in ``category_order``.
    """
    grouped = summary.groupby(categories.to_numpy())
    table = grouped.median().add_suffix("_median")
    table.insert(0, "records", grouped.size())
    return table.reindex([name for name in category_order() if name in table.index])


def mean_profiles_by_category(profiles: np.ndarray, categories: pd.Series) -> pd.DataFrame:
    """Mean profile (over time or frequency) of each category.

    Args:
        profiles: Per-record profiles, shape (records, bins).
        categories: Category of each record.

    Returns:
        Table with categories as rows (``category_order``) and bins as columns.
    """
    table = pd.DataFrame(profiles).groupby(categories.to_numpy()).mean()
    return table.reindex([name for name in category_order() if name in table.index])
