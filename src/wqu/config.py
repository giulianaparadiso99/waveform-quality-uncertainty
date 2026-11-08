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
REQUIRED_LABELS: frozenset[int] = frozenset({0, 1})


@dataclass(frozen=True)
class SubsetConfig:
    """One ESM25 subset (for example good or bad CV records)."""

    name: str
    seisbench_class: str
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
    n_waveform_examples: int
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
        n_waveform_examples=int(_require(section, "n_waveform_examples", "exploration")),
        random_seed=int(_require(section, "random_seed", "exploration")),
    )


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
        output=_build_output_config(_require(raw, "output", "root"), root),
    )


def ensure_output_dirs(output: OutputConfig) -> None:
    """Create the output directories if they do not exist.

    Args:
        output: Output configuration.
    """
    output.figures_dir.mkdir(parents=True, exist_ok=True)
    output.tables_dir.mkdir(parents=True, exist_ok=True)
