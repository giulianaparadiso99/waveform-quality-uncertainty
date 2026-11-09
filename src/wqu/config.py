"""Loading and validation of the project configuration file.

The configuration is stored as YAML (see ``configs/default.yaml``) and converted
into immutable dataclasses, so that every module receives typed, validated
parameters instead of hardcoded values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_DATA_SOURCES: tuple[str, ...] = ("seisbench", "local_csv")
VALID_WAVEFORM_SOURCES: tuple[str, ...] = ("dev_subset", "seisbench")
REQUIRED_LABELS: frozenset[int] = frozenset({0, 1})


@dataclass(frozen=True)
class SubsetConfig:
    """One ESM25 subset (for example good or bad CV records)."""

    name: str
    seisbench_class: str
    repository_name: str
    local_metadata_file: str
    label: int


@dataclass(frozen=True)
class DataConfig:
    """Where and how the ESM25 data are read."""

    source: str
    local_metadata_dir: Path
    waveform_length_s: float
    seisbench_metadata_file: str
    dataset_options: Mapping[str, Any]
    subsets: tuple[SubsetConfig, ...]


@dataclass(frozen=True)
class ColumnsConfig:
    """Names of key columns and column groups used across modules."""

    label: str
    subset: str
    subset_index: str
    station_id: str
    event_id: str
    text: tuple[str, ...]
    datetime: tuple[str, ...]
    magnitude_priority: tuple[str, ...]


@dataclass(frozen=True)
class FeatureGroup:
    """A named group of numeric and categorical features."""

    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


@dataclass(frozen=True)
class StratifiedComparison:
    """A variable whose bad rate is tabulated within the levels of a stratum."""

    variable: str
    stratum: str


@dataclass(frozen=True)
class SpatialConfig:
    """Columns and binning of the source-path and spatial overview."""

    magnitude_column: str
    distance_column: str
    year_column: str
    station_latitude_column: str
    station_longitude_column: str
    event_latitude_column: str
    event_longitude_column: str
    magnitude_start: float
    magnitude_stop: float
    magnitude_step: float
    distance_start_km: float
    distance_stop_km: float
    distance_n_bins: int
    map_extent_deg: tuple[float, float, float, float] | None
    map_margin_deg: float
    show_geography: bool


@dataclass(frozen=True)
class ExplorationConfig:
    """Parameters of the exploratory analysis."""

    feature_groups: tuple[FeatureGroup, ...]
    excluded_columns: tuple[str, ...]
    stratified_comparisons: tuple[StratifiedComparison, ...]
    n_splits: int
    target_encoding_smoothing: float
    n_quantile_bins: int
    top_categories: int
    max_discrete_levels: int
    min_stratum_count: int
    spatial: SpatialConfig
    n_waveform_examples: int
    random_seed: int


@dataclass(frozen=True)
class WaveformConfig:
    """Source and parameters of the waveform inspection."""

    source: str
    remote_url_template: str
    dev_subset_file: Path
    dev_subset_records_file: Path
    dev_subset_records_per_class: int
    dev_subset_max_records_per_event: int
    inspection_records_per_class: int | None
    batch_size: int
    zero_tolerance: float
    min_internal_gap_s: float
    spectrum_segment_s: float
    spectrum_top_sampling_rates: int
    random_seed: int


@dataclass(frozen=True)
class OutputConfig:
    """Output directories for figures and tables."""

    figures_dir: Path
    tables_dir: Path


@dataclass(frozen=True)
class ProjectConfig:
    """Complete, validated project configuration."""

    data: DataConfig
    columns: ColumnsConfig
    exploration: ExplorationConfig
    waveforms: WaveformConfig
    output: OutputConfig


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    """Return ``mapping[key]`` or raise a KeyError naming the missing entry.

    Args:
        mapping: Parsed YAML section.
        key: Required key.
        context: Section name used in the error message.

    Returns:
        The value stored under ``key``.
    """
    if key not in mapping:
        raise KeyError(f"Missing configuration entry '{context}.{key}'.")
    return mapping[key]


def _resolve_path(raw_path: str, project_root: Path) -> Path:
    """Expand the user directory and resolve relative paths against the project root.

    Args:
        raw_path: Path as written in the configuration file.
        project_root: Directory used as base for relative paths.

    Returns:
        Absolute path.
    """
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def _as_str_tuple(values: Any, context: str) -> tuple[str, ...]:
    """Convert a YAML list of strings into a tuple, validating its type.

    Args:
        values: Parsed YAML value.
        context: Entry name used in the error message.

    Returns:
        Tuple of strings (empty when the entry is null).
    """
    if values is None:
        return ()
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError(f"Configuration entry '{context}' must be a list of strings.")
    return tuple(values)


def _build_data_config(section: Mapping[str, Any], project_root: Path) -> DataConfig:
    """Build and validate the ``data`` section.

    Args:
        section: Parsed ``data`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated DataConfig.
    """
    source = _require(section, "source", "data")
    if source not in VALID_DATA_SOURCES:
        raise ValueError(f"data.source must be one of {VALID_DATA_SOURCES}, got '{source}'.")

    subsets = tuple(
        SubsetConfig(
            name=_require(entry, "name", "data.subsets"),
            seisbench_class=_require(entry, "seisbench_class", "data.subsets"),
            repository_name=_require(entry, "repository_name", "data.subsets"),
            local_metadata_file=_require(entry, "local_metadata_file", "data.subsets"),
            label=int(_require(entry, "label", "data.subsets")),
        )
        for entry in _require(section, "subsets", "data")
    )
    labels = {subset.label for subset in subsets}
    names = [subset.name for subset in subsets]
    if labels != REQUIRED_LABELS or len(subsets) != len(REQUIRED_LABELS):
        raise ValueError("data.subsets must define exactly two subsets with labels 0 and 1.")
    if len(set(names)) != len(names):
        raise ValueError("data.subsets names must be unique.")

    dataset_options = section.get("dataset_options") or {}
    if not isinstance(dataset_options, Mapping):
        raise TypeError("data.dataset_options must be a mapping.")

    return DataConfig(
        source=source,
        local_metadata_dir=_resolve_path(_require(section, "local_metadata_dir", "data"), project_root),
        waveform_length_s=float(_require(section, "waveform_length_s", "data")),
        seisbench_metadata_file=_require(section, "seisbench_metadata_file", "data"),
        dataset_options=dict(dataset_options),
        subsets=subsets,
    )


def _build_columns_config(section: Mapping[str, Any]) -> ColumnsConfig:
    """Build the ``columns`` section.

    Args:
        section: Parsed ``columns`` mapping.

    Returns:
        ColumnsConfig.
    """
    return ColumnsConfig(
        label=_require(section, "label", "columns"),
        subset=_require(section, "subset", "columns"),
        subset_index=_require(section, "subset_index", "columns"),
        station_id=_require(section, "station_id", "columns"),
        event_id=_require(section, "event_id", "columns"),
        text=_as_str_tuple(section.get("text"), "columns.text"),
        datetime=_as_str_tuple(section.get("datetime"), "columns.datetime"),
        magnitude_priority=_as_str_tuple(
            _require(section, "magnitude_priority", "columns"), "columns.magnitude_priority"
        ),
    )


def _build_spatial_config(section: Mapping[str, Any]) -> SpatialConfig:
    """Build and validate the ``exploration.spatial`` section.

    Args:
        section: Parsed ``exploration.spatial`` mapping.

    Returns:
        Validated SpatialConfig.
    """
    context = "exploration.spatial"
    magnitude_bins = _require(section, "magnitude_bins", context)
    raw_extent = section.get("map_extent_deg")
    if raw_extent is not None and (not isinstance(raw_extent, list) or len(raw_extent) != 4):
        raise ValueError(f"{context}.map_extent_deg must be null or a list of four numbers.")
    map_extent = tuple(float(value) for value in raw_extent) if raw_extent is not None else None
    distance_bins = _require(section, "distance_bins_km", context)
    spatial = SpatialConfig(
        magnitude_column=_require(section, "magnitude_column", context),
        distance_column=_require(section, "distance_column", context),
        year_column=_require(section, "year_column", context),
        station_latitude_column=_require(section, "station_latitude_column", context),
        station_longitude_column=_require(section, "station_longitude_column", context),
        event_latitude_column=_require(section, "event_latitude_column", context),
        event_longitude_column=_require(section, "event_longitude_column", context),
        magnitude_start=float(_require(magnitude_bins, "start", f"{context}.magnitude_bins")),
        magnitude_stop=float(_require(magnitude_bins, "stop", f"{context}.magnitude_bins")),
        magnitude_step=float(_require(magnitude_bins, "step", f"{context}.magnitude_bins")),
        distance_start_km=float(_require(distance_bins, "start", f"{context}.distance_bins_km")),
        distance_stop_km=float(_require(distance_bins, "stop", f"{context}.distance_bins_km")),
        distance_n_bins=int(_require(distance_bins, "n_bins", f"{context}.distance_bins_km")),
        map_extent_deg=map_extent,
        map_margin_deg=float(_require(section, "map_margin_deg", context)),
        show_geography=bool(_require(section, "show_geography", context)),
    )
    if spatial.magnitude_step <= 0 or spatial.magnitude_stop <= spatial.magnitude_start:
        raise ValueError(f"{context}.magnitude_bins must have start < stop and a positive step.")
    if spatial.distance_start_km <= 0 or spatial.distance_stop_km <= spatial.distance_start_km:
        raise ValueError(f"{context}.distance_bins_km must satisfy 0 < start < stop.")
    if spatial.distance_n_bins < 1:
        raise ValueError(f"{context}.distance_bins_km.n_bins must be positive.")
    return spatial


def _build_exploration_config(section: Mapping[str, Any]) -> ExplorationConfig:
    """Build and validate the ``exploration`` section.

    Feature columns listed among ``excluded_columns`` raise a ValueError, so that
    a column encoding the quality decision cannot be used as a feature by mistake.

    Args:
        section: Parsed ``exploration`` mapping.

    Returns:
        Validated ExplorationConfig.
    """
    feature_groups = tuple(
        FeatureGroup(
            name=_require(entry, "name", "exploration.feature_groups"),
            numeric=_as_str_tuple(entry.get("numeric"), "exploration.feature_groups.numeric"),
            categorical=_as_str_tuple(entry.get("categorical"), "exploration.feature_groups.categorical"),
        )
        for entry in _require(section, "feature_groups", "exploration")
    )
    excluded_columns = _as_str_tuple(section.get("excluded_columns"), "exploration.excluded_columns")
    all_features = [column for group in feature_groups for column in group.numeric + group.categorical]
    leaking = sorted(set(all_features) & set(excluded_columns))
    if leaking:
        raise ValueError(f"Features listed in exploration.excluded_columns: {leaking}.")
    duplicated = sorted({column for column in all_features if all_features.count(column) > 1})
    if duplicated:
        raise ValueError(f"Features listed more than once: {duplicated}.")

    comparisons = tuple(
        StratifiedComparison(
            variable=_require(entry, "variable", "exploration.stratified_comparisons"),
            stratum=_require(entry, "stratum", "exploration.stratified_comparisons"),
        )
        for entry in section.get("stratified_comparisons") or []
    )

    n_splits = int(_require(section, "n_splits", "exploration"))
    if n_splits < 2:
        raise ValueError("exploration.n_splits must be at least 2.")

    return ExplorationConfig(
        feature_groups=feature_groups,
        excluded_columns=excluded_columns,
        stratified_comparisons=comparisons,
        n_splits=n_splits,
        target_encoding_smoothing=float(_require(section, "target_encoding_smoothing", "exploration")),
        n_quantile_bins=int(_require(section, "n_quantile_bins", "exploration")),
        top_categories=int(_require(section, "top_categories", "exploration")),
        max_discrete_levels=int(_require(section, "max_discrete_levels", "exploration")),
        min_stratum_count=int(_require(section, "min_stratum_count", "exploration")),
        spatial=_build_spatial_config(_require(section, "spatial", "exploration")),
        n_waveform_examples=int(_require(section, "n_waveform_examples", "exploration")),
        random_seed=int(_require(section, "random_seed", "exploration")),
    )


def _build_waveform_config(section: Mapping[str, Any], project_root: Path) -> WaveformConfig:
    """Build and validate the ``waveforms`` section.

    Args:
        section: Parsed ``waveforms`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated WaveformConfig.
    """
    source = _require(section, "source", "waveforms")
    if source not in VALID_WAVEFORM_SOURCES:
        raise ValueError(f"waveforms.source must be one of {VALID_WAVEFORM_SOURCES}, got '{source}'.")
    template = _require(section, "remote_url_template", "waveforms")
    if "{repository_name}" not in template:
        raise ValueError("waveforms.remote_url_template must contain '{repository_name}'.")
    inspection_records = section.get("inspection_records_per_class")
    waveforms = WaveformConfig(
        source=source,
        remote_url_template=template,
        dev_subset_file=_resolve_path(_require(section, "dev_subset_file", "waveforms"), project_root),
        dev_subset_records_file=_resolve_path(_require(section, "dev_subset_records_file", "waveforms"), project_root),
        dev_subset_records_per_class=int(_require(section, "dev_subset_records_per_class", "waveforms")),
        dev_subset_max_records_per_event=int(_require(section, "dev_subset_max_records_per_event", "waveforms")),
        inspection_records_per_class=int(inspection_records) if inspection_records is not None else None,
        batch_size=int(_require(section, "batch_size", "waveforms")),
        zero_tolerance=float(_require(section, "zero_tolerance", "waveforms")),
        min_internal_gap_s=float(_require(section, "min_internal_gap_s", "waveforms")),
        spectrum_segment_s=float(_require(section, "spectrum_segment_s", "waveforms")),
        spectrum_top_sampling_rates=int(_require(section, "spectrum_top_sampling_rates", "waveforms")),
        random_seed=int(_require(section, "random_seed", "waveforms")),
    )
    positive = {
        "dev_subset_records_per_class": waveforms.dev_subset_records_per_class,
        "dev_subset_max_records_per_event": waveforms.dev_subset_max_records_per_event,
        "batch_size": waveforms.batch_size,
        "spectrum_top_sampling_rates": waveforms.spectrum_top_sampling_rates,
    }
    non_positive = [name for name, value in positive.items() if value < 1]
    if non_positive:
        raise ValueError(f"waveforms entries must be positive: {non_positive}.")
    if waveforms.zero_tolerance < 0 or waveforms.min_internal_gap_s <= 0 or waveforms.spectrum_segment_s <= 0:
        raise ValueError("waveforms.zero_tolerance must be >= 0; min_internal_gap_s and spectrum_segment_s > 0.")
    return waveforms


def _build_output_config(section: Mapping[str, Any], project_root: Path) -> OutputConfig:
    """Build the ``output`` section.

    Args:
        section: Parsed ``output`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        OutputConfig with absolute paths.
    """
    return OutputConfig(
        figures_dir=_resolve_path(_require(section, "figures_dir", "output"), project_root),
        tables_dir=_resolve_path(_require(section, "tables_dir", "output"), project_root),
    )


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> ProjectConfig:
    """Read a YAML configuration file and return a validated ProjectConfig.

    Args:
        config_path: Path to the YAML file.
        project_root: Base directory for relative paths. Defaults to the parent
            of the directory that contains the configuration file.

    Returns:
        ProjectConfig.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        KeyError, TypeError, ValueError: If entries are missing or invalid.
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    root = Path(project_root).expanduser().resolve() if project_root else config_path.parent.parent

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Configuration file {config_path} must contain a mapping.")

    return ProjectConfig(
        data=_build_data_config(_require(raw, "data", "root"), root),
        columns=_build_columns_config(_require(raw, "columns", "root")),
        exploration=_build_exploration_config(_require(raw, "exploration", "root")),
        waveforms=_build_waveform_config(_require(raw, "waveforms", "root"), root),
        output=_build_output_config(_require(raw, "output", "root"), root),
    )


def ensure_output_dirs(output: OutputConfig) -> None:
    """Create the output directories if they do not exist.

    Args:
        output: Output configuration.
    """
    output.figures_dir.mkdir(parents=True, exist_ok=True)
    output.tables_dir.mkdir(parents=True, exist_ok=True)
