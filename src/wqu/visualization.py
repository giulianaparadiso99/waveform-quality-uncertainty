"""Plotting functions for the exploratory analysis.

Each function draws on a given Matplotlib Axes (or creates a Figure when the
plot needs several panels) and returns it, so that notebooks decide layout and
saving. Colors: good and bad records use two fixed categorical colors; single
measures use one neutral color; rates use a single-hue sequential colormap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

LABEL_COLORS: Mapping[int, str] = {0: "#2a78d6", 1: "#eb6834"}
# Categorical colors for groups other than the two classes (fixed order, never cycled).
GROUP_COLORS: tuple[str, ...] = ("#1baf7a", "#eda100", "#e87ba4", "#008300")
NEUTRAL_COLOR = "#52514e"
REFERENCE_LINE_COLOR = "#9a9993"
PADDING_SHADE_COLOR = "#e6e5e0"
RATE_COLORMAP = "Oranges"
POWER_COLORMAP = "Blues"
WINDOW_TRACE_COLOR = "#2a2a28"
OUTSIDE_TRACE_COLOR = "#c3c2bc"
WINDOW_SHADE_COLOR = "#e9f1fb"
MAP_LAND_COLOR = "#f2f1ee"
MAP_EDGE_COLOR = "#9a9993"
MAP_POINT_EDGE_COLOR = "#6b6a65"
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


def plot_magnitude_distance_scatter(
    metadata: pd.DataFrame,
    distance_column: str,
    magnitude_column: str,
    label_column: str,
    label_names: Mapping[int, str],
    ax: Axes,
    point_size: float = 2.0,
    alpha: float = 0.15,
) -> Axes:
    """Scatter plot of magnitude against distance (log scale), colored by label.

    Args:
        metadata: Combined metadata.
        distance_column: Distance column (positive values).
        magnitude_column: Magnitude column.
        label_column: Label column.
        label_names: Display name of each label.
        ax: Axes to draw on.
        point_size: Marker area.
        alpha: Marker opacity (low values reveal density).

    Returns:
        The Axes.
    """
    distance = pd.to_numeric(metadata[distance_column], errors="coerce")
    magnitude = pd.to_numeric(metadata[magnitude_column], errors="coerce")
    valid = distance.gt(0) & magnitude.notna()
    for label, name in label_names.items():
        selected = valid & (metadata[label_column] == label)
        ax.scatter(
            distance[selected], magnitude[selected], s=point_size, alpha=alpha,
            color=LABEL_COLORS[label], linewidths=0, label=f"{name} (n={int(selected.sum()):,})",
        )
    ax.set_xscale("log")
    ax.set_xlabel(distance_column)
    ax.set_ylabel(magnitude_column)
    legend = ax.legend(markerscale=4)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return ax


def plot_binned_rate_2d(
    rates: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    ax: Axes,
    x_label: str,
    y_label: str,
    log_x: bool = False,
) -> Axes:
    """Map of the bad rate on a two-dimensional grid; empty cells have too few records.

    Args:
        rates: Array of shape (len(x_edges) - 1, len(y_edges) - 1) from ``binned_bad_rate_2d``.
        x_edges: Bin edges along x.
        y_edges: Bin edges along y.
        ax: Axes to draw on.
        x_label: Label of the x axis.
        y_label: Label of the y axis.
        log_x: Logarithmic x axis.

    Returns:
        The Axes.
    """
    mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(rates.T), cmap=RATE_COLORMAP, vmin=0.0, vmax=1.0)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(visible=False)
    ax.figure.colorbar(mesh, ax=ax, label="fraction of bad records")
    return ax


def create_map_axes(
    figure: Figure, subplot_position: int, extent: tuple[float, float, float, float], show_geography: bool
) -> Axes:
    """Add a geographic Axes (Plate Carree) with optional land, coastlines and borders.

    Args:
        figure: Figure receiving the Axes.
        subplot_position: Three-digit Matplotlib subplot code (e.g. 211).
        extent: (lon_min, lon_max, lat_min, lat_max).
        show_geography: Draw land, coastlines and borders (Natural Earth data).

    Returns:
        The cartopy GeoAxes.
    """
    ax = figure.add_subplot(subplot_position, projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if show_geography:
        ax.add_feature(cfeature.LAND, facecolor=MAP_LAND_COLOR, edgecolor="none")
        ax.add_feature(cfeature.COASTLINE, edgecolor=MAP_EDGE_COLOR, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor=MAP_EDGE_COLOR, linewidth=0.3)
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.3, color=MAP_EDGE_COLOR, alpha=0.5)
    gridlines.top_labels = False
    gridlines.right_labels = False
    return ax


def plot_rate_map(
    table: pd.DataFrame,
    longitude_column: str,
    latitude_column: str,
    ax: Axes,
    title: str,
    size_range: tuple[float, float] = (3.0, 120.0),
) -> Axes:
    """Points on a map colored by bad rate and sized by number of records.

    Larger groups are drawn first so that small groups remain visible.

    Args:
        table: Output of ``group_bad_rate`` with coordinate columns.
        longitude_column: Longitude column of ``table``.
        latitude_column: Latitude column of ``table``.
        ax: GeoAxes from ``create_map_axes``.
        title: Axes title.
        size_range: Marker areas of the smallest and largest group.

    Returns:
        The Axes.
    """
    ordered = table.dropna(subset=[longitude_column, latitude_column]).sort_values("n_records", ascending=False)
    root_counts = np.sqrt(ordered["n_records"].to_numpy(dtype=float))
    span = max(root_counts.max() - root_counts.min(), 1e-9)
    sizes = size_range[0] + (root_counts - root_counts.min()) / span * (size_range[1] - size_range[0])
    points = ax.scatter(
        ordered[longitude_column], ordered[latitude_column], s=sizes, c=ordered["bad_rate"],
        cmap=RATE_COLORMAP, vmin=0.0, vmax=1.0, edgecolors=MAP_POINT_EDGE_COLOR, linewidths=0.2,
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{title} (n={len(ordered):,}; marker area ~ number of records)", loc="left")
    ax.figure.colorbar(points, ax=ax, label="fraction of bad records", shrink=0.8)
    return ax


def plot_records_per_year(table: pd.DataFrame, label_names: Mapping[int, str], ax: Axes) -> Axes:
    """Number of records per year for each label.

    Args:
        table: Output of ``records_per_year_by_label``.
        label_names: Display name of each label.
        ax: Axes to draw on.

    Returns:
        The Axes.
    """
    for label, name in label_names.items():
        if label in table.columns:
            ax.step(table.index, table[label], where="mid", linewidth=LINE_WIDTH,
                    color=LABEL_COLORS[label], label=name)
    ax.set_xlabel("year")
    ax.set_ylabel("number of records")
    ax.legend()
    return ax


def plot_mean_spectra(mean_log_psd: pd.DataFrame, counts: pd.Series, ax: Axes) -> Axes:
    """Mean power spectra per original sampling rate, with each Nyquist frequency marked.

    Args:
        mean_log_psd: Mean log10 PSD, frequencies x groups (original sampling rates in Hz).
        counts: Number of averaged spectra per group.
        ax: Axes to draw on.

    Returns:
        The Axes.
    """
    if mean_log_psd.shape[1] > len(GROUP_COLORS):
        raise ValueError(f"At most {len(GROUP_COLORS)} groups can be drawn, got {mean_log_psd.shape[1]}.")
    positive = mean_log_psd.index > 0
    for color, group in zip(GROUP_COLORS, mean_log_psd.columns):
        ax.plot(
            mean_log_psd.index[positive], mean_log_psd.loc[positive, group], color=color,
            linewidth=LINE_WIDTH, label=f"{group:g} Hz (n={int(counts[group]):,})",
        )
        ax.axvline(group / 2.0, color=color, linewidth=1.0, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("frequency [Hz] (dotted: original Nyquist frequency)")
    ax.set_ylabel("mean log10 PSD")
    ax.legend(title="original sampling rate")
    return ax


def plot_window_overlay(
    trace: np.ndarray,
    sampling_rate_hz: float,
    window_start: int,
    window_stop: int,
    components: Sequence[str],
    title: str,
) -> Figure:
    """Plot each component of a stored trace with the analysis window highlighted.

    Samples outside the window (removed by the cut) are drawn in light grey and
    samples inside it in dark grey, on top of each other, so that the cut can be
    checked visually.

    Args:
        trace: Stored trace, shape (components, samples).
        sampling_rate_hz: Sampling rate of the trace.
        window_start: First sample of the window (may be negative).
        window_stop: Sample after the last one of the window (may exceed the trace).
        components: Component names (one per row).
        title: Figure title.

    Returns:
        The Figure.
    """
    time_s = np.arange(trace.shape[-1]) / sampling_rate_hz
    inside = (np.arange(trace.shape[-1]) >= window_start) & (np.arange(trace.shape[-1]) < window_stop)
    figure, axes = plt.subplots(len(components), 1, figsize=(10, 1.8 * len(components)), sharex=True)
    for ax, component, values in zip(np.atleast_1d(axes), components, trace):
        ax.axvspan(window_start / sampling_rate_hz, window_stop / sampling_rate_hz, color=WINDOW_SHADE_COLOR, linewidth=0)
        ax.plot(time_s, values, color=OUTSIDE_TRACE_COLOR, linewidth=0.6)
        ax.plot(time_s, np.where(inside, values, np.nan), color=WINDOW_TRACE_COLOR, linewidth=0.6)
        ax.set_ylabel(f"acceleration {component}")
        ax.set_xlim(time_s[0], time_s[-1])
    np.atleast_1d(axes)[-1].set_xlabel("time from trace start [s] (dark, shaded: analysis window; light: removed)")
    figure.suptitle(title, x=0.01, ha="left")
    figure.tight_layout()
    return figure


def plot_time_frequency(
    image_db: np.ndarray,
    frequency_edges_hz: np.ndarray,
    times_s: np.ndarray,
    ax: Axes,
    value_range: tuple[float, float],
) -> object:
    """Draw one time-frequency image (decibels) with a logarithmic frequency axis.

    Args:
        image_db: Array of shape (frequencies, times).
        frequency_edges_hz: Band edges (frequencies + 1 values).
        times_s: Time-bin centres (equally spaced).
        ax: Axes to draw on.
        value_range: Colour scale limits (dB).

    Returns:
        The QuadMesh, for a shared colour bar.
    """
    step = times_s[1] - times_s[0]
    time_edges = np.concatenate((times_s - step / 2, [times_s[-1] + step / 2]))
    mesh = ax.pcolormesh(time_edges, frequency_edges_hz, image_db, cmap=POWER_COLORMAP,
                         vmin=value_range[0], vmax=value_range[1], shading="flat")
    ax.set_yscale("log")
    ax.grid(visible=False)
    return mesh


def plot_representation_comparison(
    window_waveform: np.ndarray,
    waveform_rate_hz: float,
    spectrogram_db: np.ndarray,
    scalogram_db: np.ndarray,
    frequency_edges_hz: np.ndarray,
    times_s: np.ndarray,
    components: Sequence[str],
    title: str,
) -> Figure:
    """Waveform, spectrogram and scalogram of the analysis window, one row per component.

    Each representation uses one colour scale for its three components (2nd to
    99.5th percentile of its values), so that differences between components stay visible.

    Args:
        window_waveform: Resampled waveform of the window, shape (components, samples).
        waveform_rate_hz: Sampling rate of ``window_waveform``.
        spectrogram_db: Array of shape (components, frequencies, times).
        scalogram_db: Array of shape (components, frequencies, times).
        frequency_edges_hz: Band edges of the grid.
        times_s: Time-bin centres of the grid.
        components: Component names.
        title: Figure title.

    Returns:
        The Figure.
    """
    figure, axes = plt.subplots(
        len(components), 3, figsize=(16, 2.8 * len(components)), sharex=True, squeeze=False, layout="constrained"
    )
    waveform_time_s = np.arange(window_waveform.shape[-1]) / waveform_rate_hz
    representations = {"spectrogram [dB]": spectrogram_db, "scalogram [dB]": scalogram_db}
    meshes = {}
    for row, component in enumerate(components):
        axes[row, 0].plot(waveform_time_s, window_waveform[row], color=WINDOW_TRACE_COLOR, linewidth=0.5)
        axes[row, 0].set_ylabel(f"acceleration {component}")
        for column, (name, values) in enumerate(representations.items(), start=1):
            value_range = tuple(np.percentile(values, [2.0, 99.5]))
            meshes[name] = plot_time_frequency(values[row], frequency_edges_hz, times_s, axes[row, column], value_range)
            axes[row, column].set_ylabel("frequency [Hz]")
    for column, name in enumerate(representations, start=1):
        axes[0, column].set_title(name, loc="left")
        figure.colorbar(meshes[name], ax=axes[:, column], shrink=0.9, pad=0.02)
    axes[0, 0].set_title("waveform (analysis window, resampled)", loc="left")
    for column in range(3):
        axes[-1, column].set_xlabel("time from window start [s]")
    figure.suptitle(title, x=0.01, ha="left")
    return figure


def plot_value_distributions(
    values_by_name: Mapping[str, np.ndarray],
    ax: Axes,
    x_label: str,
    n_bins: int = 100,
    quantile_range: tuple[float, float] = (0.0, 1.0),
) -> Axes:
    """Normalized histograms of several sets of values on a common set of bins.

    Args:
        values_by_name: Mapping from name to values (non-finite values are ignored).
        ax: Axes to draw on.
        x_label: Label of the x axis.
        n_bins: Number of bins.
        quantile_range: The x range spans these quantiles of the pooled values,
            so that a few extreme values do not hide the bulk of the distribution.

    Returns:
        The Axes.
    """
    if len(values_by_name) > len(GROUP_COLORS):
        raise ValueError(f"At most {len(GROUP_COLORS)} sets can be drawn.")
    finite = {name: values[np.isfinite(values)] for name, values in values_by_name.items()}
    pooled = np.concatenate(list(finite.values()))
    low, high = (float(value) for value in np.quantile(pooled, quantile_range))
    edges = np.linspace(low, high, n_bins + 1)
    for color, (name, values) in zip(GROUP_COLORS, finite.items()):
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=LINE_WIDTH, color=color, label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel("density")
    ax.legend()
    return ax


WINDOW_MARKER_STYLES: Mapping[str, str] = {"suggested": "-", "configured": "--"}
REJECTED_HATCH = "///"
WINDOW_LENGTH_COLORMAP = "Blues"


def _grid_position(values: np.ndarray, value: float) -> int:
    """Position of a value in a sorted axis of the window grid.

    Args:
        values: Axis values.
        value: Value to locate.

    Returns:
        Position in ``values``.

    Raises:
        KeyError: If the value is not on the axis.
    """
    matches = np.flatnonzero(np.isclose(values, value))
    if matches.size == 0:
        raise KeyError(f"Value {value} is not on the grid axis {values.tolist()}.")
    return int(matches[0])


def plot_window_grid_heatmap(
    summary: pd.DataFrame,
    value_column: str,
    ax: Axes,
    title: str,
    colormap: str,
    value_format: str,
    marked_windows: Mapping[str, tuple[float, float]],
) -> Axes:
    """Heatmap of a window-search measure over (pre_pga_s, post_pga_s).

    Rejected windows (``accepted`` false) are hatched; the windows in
    ``marked_windows`` are outlined (solid for ``suggested``, dashed otherwise).

    Args:
        summary: Output of ``summarize_window_grid``.
        value_column: Column to show.
        ax: Axes to draw on.
        title: Axes title.
        colormap: Matplotlib colormap name.
        value_format: Format of the cell annotations (e.g. ``.3f``).
        marked_windows: Mapping from a name to (pre_pga_s, post_pga_s).

    Returns:
        The Axes.
    """
    table = summary.pivot(index="pre_pga_s", columns="post_pga_s", values=value_column)
    accepted = summary.pivot(index="pre_pga_s", columns="post_pga_s", values="accepted").astype(bool)
    pre_values, post_values = table.index.to_numpy(), table.columns.to_numpy()
    image = ax.imshow(table.to_numpy(), cmap=colormap, aspect="auto", origin="lower")
    ax.figure.colorbar(image, ax=ax, label=value_column)
    values = table.to_numpy()
    threshold = np.nanmean(values)
    for row in range(len(pre_values)):
        for column in range(len(post_values)):
            ax.text(column, row, format(values[row, column], value_format), ha="center", va="center", fontsize=7,
                    color="white" if values[row, column] > threshold else WINDOW_TRACE_COLOR)
            if not accepted.iloc[row, column]:
                ax.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, hatch=REJECTED_HATCH,
                                           edgecolor=REFERENCE_LINE_COLOR, linewidth=0))
    for name, (pre, post) in marked_windows.items():
        if np.isclose(pre_values, pre).any() and np.isclose(post_values, post).any():
            ax.add_patch(plt.Rectangle(
                (_grid_position(post_values, post) - 0.5, _grid_position(pre_values, pre) - 0.5), 1, 1, fill=False,
                edgecolor="black", linewidth=2.0, linestyle=WINDOW_MARKER_STYLES.get(name, ":"), label=name,
            ))
    ax.set_xticks(range(len(post_values)), [f"{value:g}" for value in post_values])
    ax.set_yticks(range(len(pre_values)), [f"{value:g}" for value in pre_values])
    ax.set_xlabel("post_pga_s [s]")
    ax.set_ylabel("pre_pga_s [s]")
    ax.set_title(title, loc="left")
    ax.grid(False)
    return ax


def plot_window_tradeoff(
    summary: pd.DataFrame,
    pareto: pd.Series,
    max_padding_separation: float,
    marked_windows: Mapping[str, tuple[float, float]],
    ax: Axes,
) -> Axes:
    """Padding separation against energy score for every candidate window.

    Points are coloured by window length; the Pareto front is joined by a line,
    the dashed vertical line is the accepted padding separation, and the windows
    in ``marked_windows`` are circled and labelled.

    Args:
        summary: Output of ``summarize_window_grid``.
        pareto: Boolean Series marking the Pareto front (see ``pareto_front``).
        max_padding_separation: Accepted padding separation.
        marked_windows: Mapping from a name to (pre_pga_s, post_pga_s).
        ax: Axes to draw on.

    Returns:
        The Axes.
    """
    points = ax.scatter(summary["padding_separation"], summary["energy_score"], c=summary["window_length_s"],
                        cmap=WINDOW_LENGTH_COLORMAP, edgecolors=NEUTRAL_COLOR, linewidths=0.5, s=40)
    ax.figure.colorbar(points, ax=ax, label="window length [s]")
    front = summary[pareto].sort_values("padding_separation")
    (front_line,) = ax.plot(front["padding_separation"], front["energy_score"], color=NEUTRAL_COLOR, linewidth=1.0,
                            label="Pareto front")
    threshold_line = ax.axvline(max_padding_separation, color=REFERENCE_LINE_COLOR, linestyle="--", linewidth=1.0,
                                label="max_padding_separation")
    for name, (pre, post) in marked_windows.items():
        match = summary[np.isclose(summary["pre_pga_s"], pre) & np.isclose(summary["post_pga_s"], post)]
        if match.empty:
            continue
        row = match.iloc[0]
        ax.scatter(row["padding_separation"], row["energy_score"], s=160, facecolors="none", edgecolors="black",
                   linewidths=1.5, linestyle=WINDOW_MARKER_STYLES.get(name, ":"))
        ax.annotate(f"{name} ({pre:g}, {post:g})", (row["padding_separation"], row["energy_score"]),
                    xytext=(8, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("padding separation (max of AUC, 1 - AUC)")
    ax.set_ylabel("energy score")
    ax.legend(handles=[front_line, threshold_line], loc="best")
    return ax
