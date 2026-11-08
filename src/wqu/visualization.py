"""Plotting functions for the exploratory analysis.

Each function draws on a given Matplotlib Axes (or creates a Figure when the
plot needs several panels) and returns it, so that notebooks decide layout and
saving. Colors: good and bad records use two fixed categorical colors; single
measures use one neutral color; rates use a single-hue sequential colormap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

LABEL_COLORS: Mapping[int, str] = {0: "#2a78d6", 1: "#eb6834"}
NEUTRAL_COLOR = "#52514e"
REFERENCE_LINE_COLOR = "#9a9993"
PADDING_SHADE_COLOR = "#e6e5e0"
RATE_COLORMAP = "Oranges"
LINE_WIDTH = 2.0
DEFAULT_DPI = 300


def apply_plot_style() -> None:
    """Set a light, recessive style for axes and grids."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DEFAULT_DPI,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": REFERENCE_LINE_COLOR,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#ececea",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(figure: Figure, path: Path) -> Path:
    """Save a figure, creating the parent directory if needed.

    Args:
        figure: Figure to save.
        path: Output file (format from the extension).

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    return path


def _histogram_bins(values: pd.Series, n_bins: int, log_scale: bool) -> np.ndarray:
    """Compute common histogram bin edges for all classes.

    Args:
        values: Positive values when ``log_scale`` is True.
        n_bins: Number of bins.
        log_scale: Use logarithmically spaced edges.

    Returns:
        Bin edges.
    """
    low, high = float(values.min()), float(values.max())
    if log_scale:
        return np.logspace(np.log10(low), np.log10(high), n_bins + 1)
    return np.linspace(low, high, n_bins + 1)


def plot_numeric_distribution_by_label(
    metadata: pd.DataFrame,
    column: str,
    label_column: str,
    label_names: Mapping[int, str],
    ax: Axes,
    n_bins: int = 50,
    log_scale: bool = False,
    quantile_range: tuple[float, float] | None = None,
) -> Axes:
    """Overlay the normalized histograms of a numeric column for each label.

    Args:
        metadata: Combined metadata.
        column: Numeric column.
        label_column: Label column.
        label_names: Display name of each label.
        ax: Axes to draw on.
        n_bins: Number of bins.
        log_scale: Logarithmic x axis (non-positive values are dropped).
        quantile_range: If given, only values between these quantiles of the
            pooled data are shown, so that extreme outliers do not hide the bulk.

    Returns:
        The Axes.
    """
    values = pd.to_numeric(metadata[column], errors="coerce")
    valid = values.notna() & (values > 0 if log_scale else True)
    if quantile_range is not None:
        low, high = values[valid].quantile(list(quantile_range))
        valid &= values.between(low, high)
    if not valid.any():
        raise ValueError(f"Column '{column}' has no values to plot.")
    edges = _histogram_bins(values[valid], n_bins, log_scale)
    for label, name in label_names.items():
        selected = values[valid & (metadata[label_column] == label)]
        ax.hist(
            selected,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=LINE_WIDTH,
            color=LABEL_COLORS[label],
            label=f"{name} (n={len(selected):,})",
        )
    if log_scale:
        ax.set_xscale("log")
    ax.set_xlabel(column)
    ax.set_ylabel("density")
    ax.legend()
    return ax


def plot_category_bad_rate(table: pd.DataFrame, overall_bad_rate: float, ax: Axes) -> Axes:
    """Horizontal bars of the bad rate per category, annotated with record counts.

    Args:
        table: Output of ``categorical_distribution_by_label``.
        overall_bad_rate: Bad rate of the whole dataset, drawn as reference line.
        ax: Axes to draw on.

    Returns:
        The Axes.
    """
    ordered = table.iloc[::-1]
    positions = np.arange(len(ordered))
    ax.barh(positions, ordered["bad_rate"], color=LABEL_COLORS[1], height=0.7)
    ax.axvline(overall_bad_rate, color=REFERENCE_LINE_COLOR, linewidth=1.2, linestyle="--")
    ax.set_yticks(positions, ordered.index.astype(str))
    for position, (rate, count) in enumerate(zip(ordered["bad_rate"], ordered["n_records"])):
        ax.text(rate + 0.01, position, f"n={count:,}", va="center", fontsize=8, color=NEUTRAL_COLOR)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("fraction of bad records")
    ax.set_ylabel(table.index.name)
    ax.grid(axis="y", visible=False)
    return ax


def plot_univariate_auc(auc_table: pd.DataFrame, group_order: Sequence[str]) -> Figure:
    """Bar chart of the univariate separation of each feature, one panel per group.

    Args:
        auc_table: Output of ``univariate_auc_table``.
        group_order: Order of the feature groups (one panel each).

    Returns:
        The Figure.
    """
    heights = [max(1, int((auc_table["group"] == group).sum())) for group in group_order]
    figure, axes = plt.subplots(
        len(group_order), 1, figsize=(7, 0.35 * sum(heights) + 1.2 * len(group_order)),
        gridspec_kw={"height_ratios": heights}, sharex=True, squeeze=False,
    )
    for ax, group in zip(axes[:, 0], group_order):
        rows = auc_table[auc_table["group"] == group].sort_values("separation")
        positions = np.arange(len(rows))
        ax.barh(positions, rows["separation"] - 0.5, left=0.5, color=NEUTRAL_COLOR, height=0.6)
        ax.set_yticks(positions, rows["feature"])
        for position, value in enumerate(rows["separation"]):
            ax.text(value + 0.005, position, f"{value:.2f}", va="center", fontsize=8, color=NEUTRAL_COLOR)
        ax.axvline(0.5, color=REFERENCE_LINE_COLOR, linewidth=1.0)
        ax.set_title(group, loc="left")
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlim(0.5, 1.0)
    axes[-1, 0].set_xlabel("univariate separation  max(AUC, 1 - AUC)")
    figure.tight_layout()
    return figure


def plot_stratified_bad_rate(rates: pd.DataFrame, counts: pd.DataFrame, ax: Axes) -> Axes:
    """Heatmap of bad rates within strata, annotated with rate and record count.

    Args:
        rates: Bad-rate table from ``stratified_bad_rate`` (missing = too few records).
        counts: Record-count table from ``stratified_bad_rate``.
        ax: Axes to draw on.

    Returns:
        The Axes.
    """
    image = ax.imshow(rates.to_numpy(dtype=float), cmap=RATE_COLORMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(rates.shape[1]), rates.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(rates.shape[0]), rates.index.astype(str))
    for row in range(rates.shape[0]):
        for column in range(rates.shape[1]):
            rate = rates.iat[row, column]
            if pd.notna(rate):
                text_color = "white" if rate > 0.6 else "black"
                ax.text(column, row, f"{rate:.2f}\n{int(counts.iat[row, column]):,}",
                        ha="center", va="center", fontsize=7, color=text_color)
    ax.set_xlabel(rates.columns.name)
    ax.set_ylabel(rates.index.name)
    ax.grid(visible=False)
    ax.figure.colorbar(image, ax=ax, label="fraction of bad records")
    return ax


def plot_waveform_with_padding(
    waveform: np.ndarray,
    sampling_rate_hz: float,
    signal_windows: Mapping[str, tuple[float, float]],
    title: str,
) -> Figure:
    """Plot the components of a trace, shading the zero-padded intervals.

    Args:
        waveform: Array of shape (components, samples).
        sampling_rate_hz: Sampling rate of the array.
        signal_windows: Interval with recorded data per component, in seconds.
        title: Figure title.

    Returns:
        The Figure.
    """
    components = list(signal_windows)
    if waveform.shape[0] != len(components):
        raise ValueError("Number of waveform components and signal windows differ.")
    time_s = np.arange(waveform.shape[1]) / sampling_rate_hz
    figure, axes = plt.subplots(len(components), 1, figsize=(10, 1.8 * len(components)), sharex=True)
    for ax, component, trace in zip(np.atleast_1d(axes), components, waveform):
        start_s, end_s = signal_windows[component]
        ax.axvspan(time_s[0], start_s, color=PADDING_SHADE_COLOR, linewidth=0)
        ax.axvspan(end_s, time_s[-1], color=PADDING_SHADE_COLOR, linewidth=0)
        ax.plot(time_s, trace, color=NEUTRAL_COLOR, linewidth=0.6)
        ax.set_ylabel(f"acceleration {component}")
        ax.set_xlim(time_s[0], time_s[-1])
    np.atleast_1d(axes)[-1].set_xlabel("time from trace start [s] (shaded: zero padding)")
    figure.suptitle(title, x=0.01, ha="left")
    figure.tight_layout()
    return figure
