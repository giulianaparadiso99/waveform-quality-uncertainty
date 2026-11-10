"""Loading and validation of the project configuration file.

The configuration is stored as YAML (see ``configs/default.yaml``) and converted
into immutable dataclasses, so that every module receives typed, validated
parameters instead of hardcoded values.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_DATA_SOURCES: tuple[str, ...] = ("seisbench", "local_csv")
VALID_WAVEFORM_SOURCES: tuple[str, ...] = ("dev_subset", "seisbench")
VALID_ENERGY_CRITERIA: tuple[str, ...] = ("quantile", "share_above")
VALID_ANCHOR_METHODS: tuple[str, ...] = ("pga", "envelope", "arias", "theoretical", "hybrid")
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
class WindowSearchConfig:
    """Grid search over the analysis window around the PGA.

    Attributes:
        pre_anchor_s: Candidate lengths before the PGA.
        post_anchor_s: Candidate lengths after the PGA.
        max_padding_separation: Largest accepted separation (max of AUC and 1 - AUC)
            between the classes of the zero padding contained in the window, over the
            univariate measures and the padding classifier.
        exclude_empty_components: Exclude from the leakage tests the records with a
            component entirely without data.
        classifier_n_splits: Folds (grouped by event) of the padding classifier.
        classifier_mask_bins: Time bins of the window in the padding mask seen by the classifier.
        classifier_random_seed: Random seed of the folds and of the classifier.
        classifier_n_permutations: Label permutations used to estimate the separation
            reached by the classifier when the padding carries no class information.
        energy_reference_label: Label of the records on which the energy kept is measured.
        energy_criterion: ``quantile`` (a quantile of the energy fraction kept) or
            ``share_above`` (share of records keeping at least ``min_energy_fraction``).
        energy_quantile: Quantile used by the ``quantile`` criterion.
        min_energy_fraction: Threshold used by the ``share_above`` criterion.
        score_tolerance: Feasible windows whose energy score is within this amount of the
            best are considered equivalent; the shortest of them is selected.
    """

    pre_anchor_s: tuple[float, ...]
    post_anchor_s: tuple[float, ...]
    max_padding_separation: float
    exclude_empty_components: bool
    classifier_n_splits: int
    classifier_mask_bins: int
    classifier_random_seed: int
    classifier_n_permutations: int
    energy_reference_label: int
    energy_criterion: str
    energy_quantile: float
    min_energy_fraction: float
    score_tolerance: float


@dataclass(frozen=True)
class AnchoringConfig:
    """Comparison of the methods that place the analysis window.

    Attributes:
        methods: Methods to compare (see ``wqu.preprocessing.anchoring``).
        envelope_window_s: Moving-average length of the energy envelope.
        arias_fraction: Fraction of the cumulative energy defining the Arias anchor.
        distance_column: Metadata column with the source-station distance in km.
        phase_velocity_km_s: Velocity used for the theoretical arrival.
        hybrid_search_half_width_s: Half-width of the interval around the theoretical
            arrival in which the hybrid method looks for the envelope maximum.
        agreement_tolerance_s: Two anchors disagree when they differ by more than this.
        n_examples_per_class: Records per class shown where the configured anchor and the PGA disagree.
        gallery_records_per_class: Random records per class in the gallery of windows of every method.
    """

    methods: tuple[str, ...]
    envelope_window_s: float
    arias_fraction: float
    distance_column: str
    phase_velocity_km_s: float
    hybrid_search_half_width_s: float
    agreement_tolerance_s: float
    n_examples_per_class: int
    gallery_records_per_class: int


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of the waveform-to-image preprocessing."""

    anchor_method: str
    pre_anchor_s: float
    post_anchor_s: float
    context_margin_s: float
    taper_fraction: float
    target_sampling_rate_hz: float
    antialias_corner_hz: float
    antialias_order: int
    min_frequency_hz: float
    max_frequency_hz: float
    n_frequencies: int
    n_times: int
    stft_window_cycles: float
    stft_window_type: str
    morlet_w0: float
    reference_power: float
    floor_power: float
    window_search: WindowSearchConfig
    anchoring: AnchoringConfig
    n_distribution_records: int
    distribution_plot_margin_db: float
    n_png_records_per_class: int
    output_dir: Path
    figures_dir: Path
    tables_dir: Path

    @property
    def window_length_s(self) -> float:
        """Length of the analysis window in seconds."""
        return self.pre_anchor_s + self.post_anchor_s

    @property
    def stft_window_s(self) -> float:
        """Length of the STFT window in seconds."""
        return self.stft_window_cycles / self.min_frequency_hz

    @property
    def morlet_half_width_s(self) -> float:
        """Standard deviation of the Morlet envelope at the lowest frequency, in seconds."""
        return self.morlet_w0 / (2.0 * math.pi * self.min_frequency_hz)


@dataclass(frozen=True)
class SplitConfig:
    """Training/validation split of ESM25.

    Attributes:
        validation_fraction: Approximate share of records in the validation set; the
            split takes one fold of ``round(1 / validation_fraction)`` folds.
        random_seed: Random seed of the fold assignment.
        output_file: CSV file storing the split of every record.
        tables_dir: Output directory for the summary tables.
    """

    validation_fraction: float
    random_seed: int
    output_file: Path
    tables_dir: Path

    @property
    def n_folds(self) -> int:
        """Number of folds whose first one is the validation set."""
        return round(1.0 / self.validation_fraction)


@dataclass(frozen=True)
class ModelConfig:
    """Convolutional-recurrent network adapted from CRED (Mousavi et al., 2019).

    Attributes:
        conv_filters: Filters of each convolutional stage.
        conv_kernels: Kernel size of the strided convolution opening each stage; the two
            convolutions of the residual block of the stage use ``kernel - 2``.
        recurrent_units: Hidden units per direction of the bidirectional LSTM layers.
        recurrent_layers: Number of bidirectional LSTM layers (residual from the second on).
        head_units: Units of the hidden dense layer of the classification head.
        conv_dropout: Channel dropout rate after each convolutional stage.
        recurrent_dropout: Dropout rate after each LSTM layer.
        head_dropout: Dropout rate in the classification head.
        mc_samples: Stochastic forward passes used for MC dropout predictions.
        grad_cam_mc_samples: Stochastic passes used for each Grad-CAM map.
        grad_cam_records_per_class: Records per class shown in the Grad-CAM check.
        figures_dir: Output directory for the figures of the model checks.
        tables_dir: Output directory for the tables of the model checks.
    """

    conv_filters: tuple[int, ...]
    conv_kernels: tuple[int, ...]
    recurrent_units: int
    recurrent_layers: int
    head_units: int
    conv_dropout: float
    recurrent_dropout: float
    head_dropout: float
    mc_samples: int
    grad_cam_mc_samples: int
    grad_cam_records_per_class: int
    figures_dir: Path
    tables_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Training of the classifier.

    Attributes:
        representation: ``spectrogram`` or ``scalogram``.
        feature_source: Waveform source whose feature file is used (``dev_subset`` or ``seisbench``).
        batch_size: Records per batch.
        learning_rate: Initial learning rate of Adam.
        weight_decay: Weight decay of Adam.
        max_epochs: Largest number of epochs.
        lr_factor: Factor applied to the learning rate when the validation loss stalls.
        lr_patience: Epochs without improvement before reducing the learning rate.
        early_stopping_patience: Epochs without improvement before stopping.
        num_workers: Data-loading worker processes (0 = main process).
        device: ``auto`` (Apple GPU, then CUDA, then CPU), ``cpu``, ``mps`` or ``cuda``.
        random_seed: Seed of weights, batch order and dropout.
        calibration_bins: Bins of the expected calibration error.
        runs_dir: Directory where every run gets its own folder.
    """

    representation: str
    feature_source: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    lr_factor: float
    lr_patience: int
    early_stopping_patience: int
    num_workers: int
    device: str
    random_seed: int
    calibration_bins: int
    runs_dir: Path


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
    preprocessing: PreprocessingConfig
    split: SplitConfig
    model: ModelConfig
    training: TrainingConfig
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


def _as_positive_float_tuple(values: Any, context: str) -> tuple[float, ...]:
    """Convert a YAML list of numbers into a sorted tuple of unique non-negative floats.

    Args:
        values: Parsed YAML value.
        context: Entry name used in the error message.

    Returns:
        Sorted tuple of unique floats.
    """
    if not isinstance(values, list) or not values:
        raise TypeError(f"Configuration entry '{context}' must be a non-empty list of numbers.")
    converted = sorted({float(value) for value in values})
    if converted[0] < 0:
        raise ValueError(f"Configuration entry '{context}' must contain non-negative values.")
    return tuple(converted)


def _build_window_search_config(section: Mapping[str, Any]) -> WindowSearchConfig:
    """Build and validate the ``preprocessing.window_search`` section.

    Args:
        section: Parsed ``window_search`` mapping.

    Returns:
        Validated WindowSearchConfig.
    """
    context = "preprocessing.window_search"
    leakage = _require(section, "leakage", context)
    classifier = _require(leakage, "classifier", f"{context}.leakage")
    energy = _require(section, "energy", context)
    config = WindowSearchConfig(
        pre_anchor_s=_as_positive_float_tuple(_require(section, "pre_anchor_s", context), f"{context}.pre_anchor_s"),
        post_anchor_s=_as_positive_float_tuple(_require(section, "post_anchor_s", context), f"{context}.post_anchor_s"),
        max_padding_separation=float(_require(leakage, "max_separation", f"{context}.leakage")),
        exclude_empty_components=bool(_require(leakage, "exclude_empty_components", f"{context}.leakage")),
        classifier_n_splits=int(_require(classifier, "n_splits", f"{context}.leakage.classifier")),
        classifier_mask_bins=int(_require(classifier, "mask_bins", f"{context}.leakage.classifier")),
        classifier_random_seed=int(_require(classifier, "random_seed", f"{context}.leakage.classifier")),
        classifier_n_permutations=int(_require(classifier, "n_permutations", f"{context}.leakage.classifier")),
        energy_reference_label=int(_require(energy, "reference_label", f"{context}.energy")),
        energy_criterion=str(_require(energy, "criterion", f"{context}.energy")),
        energy_quantile=float(_require(energy, "quantile", f"{context}.energy")),
        min_energy_fraction=float(_require(energy, "min_fraction", f"{context}.energy")),
        score_tolerance=float(_require(section, "score_tolerance", context)),
    )
    problems = []
    if not 0.5 <= config.max_padding_separation <= 1.0:
        problems.append("leakage.max_separation must be in [0.5, 1]")
    if config.classifier_n_splits < 2:
        problems.append("leakage.classifier.n_splits must be at least 2")
    if config.classifier_mask_bins < 1:
        problems.append("leakage.classifier.mask_bins must be at least 1")
    if config.classifier_n_permutations < 1:
        problems.append("leakage.classifier.n_permutations must be at least 1")
    if config.energy_reference_label not in REQUIRED_LABELS:
        problems.append(f"energy.reference_label must be one of {sorted(REQUIRED_LABELS)}")
    if config.energy_criterion not in VALID_ENERGY_CRITERIA:
        problems.append(f"energy.criterion must be one of {VALID_ENERGY_CRITERIA}")
    if not 0.0 <= config.energy_quantile <= 1.0:
        problems.append("energy.quantile must be in [0, 1]")
    if not 0.0 < config.min_energy_fraction <= 1.0:
        problems.append("energy.min_fraction must be in (0, 1]")
    if config.score_tolerance < 0:
        problems.append("score_tolerance must be non-negative")
    if min(config.post_anchor_s) <= 0:
        problems.append("post_anchor_s values must be positive")
    if problems:
        raise ValueError(f"Invalid {context} configuration: " + "; ".join(problems) + ".")
    return config


def _build_anchoring_config(section: Mapping[str, Any]) -> AnchoringConfig:
    """Build and validate the ``preprocessing.anchoring`` section.

    Args:
        section: Parsed ``anchoring`` mapping.

    Returns:
        Validated AnchoringConfig.
    """
    context = "preprocessing.anchoring"
    theoretical = _require(section, "theoretical", context)
    config = AnchoringConfig(
        methods=_as_str_tuple(_require(section, "methods", context), f"{context}.methods"),
        envelope_window_s=float(_require(section, "envelope_window_s", context)),
        arias_fraction=float(_require(section, "arias_fraction", context)),
        distance_column=str(_require(theoretical, "distance_column", f"{context}.theoretical")),
        phase_velocity_km_s=float(_require(theoretical, "phase_velocity_km_s", f"{context}.theoretical")),
        hybrid_search_half_width_s=float(_require(section, "hybrid_search_half_width_s", context)),
        agreement_tolerance_s=float(_require(section, "agreement_tolerance_s", context)),
        n_examples_per_class=int(_require(section, "n_examples_per_class", context)),
        gallery_records_per_class=int(_require(section, "gallery_records_per_class", context)),
    )
    problems = []
    unknown = sorted(set(config.methods) - set(VALID_ANCHOR_METHODS))
    if unknown or not config.methods:
        problems.append(f"methods must be a non-empty subset of {VALID_ANCHOR_METHODS}, got unknown {unknown}")
    if config.envelope_window_s <= 0:
        problems.append("envelope_window_s must be positive")
    if not 0.0 < config.arias_fraction < 1.0:
        problems.append("arias_fraction must be in (0, 1)")
    if config.phase_velocity_km_s <= 0:
        problems.append("theoretical.phase_velocity_km_s must be positive")
    if config.hybrid_search_half_width_s <= 0 or config.agreement_tolerance_s <= 0:
        problems.append("hybrid_search_half_width_s and agreement_tolerance_s must be positive")
    if config.n_examples_per_class < 0 or config.gallery_records_per_class < 0:
        problems.append("n_examples_per_class and gallery_records_per_class must be non-negative")
    if problems:
        raise ValueError(f"Invalid {context} configuration: " + "; ".join(problems) + ".")
    return config


def _build_preprocessing_config(section: Mapping[str, Any], project_root: Path) -> PreprocessingConfig:
    """Build and validate the ``preprocessing`` section.

    Checks that the frequency band fits below the anti-alias corner and the new
    Nyquist frequency, and that the context margin covers the STFT window and
    the Morlet wavelet at the lowest frequency.

    Args:
        section: Parsed ``preprocessing`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated PreprocessingConfig.
    """
    context = "preprocessing"
    window = _require(section, "window", context)
    resampling = _require(section, "resampling", context)
    grid = _require(section, "grid", context)
    stft = _require(section, "stft", context)
    cwt = _require(section, "cwt", context)
    decibel = _require(section, "decibel", context)
    config = PreprocessingConfig(
        anchor_method=str(_require(window, "anchor", f"{context}.window")),
        pre_anchor_s=float(_require(window, "pre_anchor_s", f"{context}.window")),
        post_anchor_s=float(_require(window, "post_anchor_s", f"{context}.window")),
        context_margin_s=float(_require(window, "context_margin_s", f"{context}.window")),
        taper_fraction=float(_require(window, "taper_fraction", f"{context}.window")),
        target_sampling_rate_hz=float(_require(resampling, "target_sampling_rate_hz", f"{context}.resampling")),
        antialias_corner_hz=float(_require(resampling, "antialias_corner_hz", f"{context}.resampling")),
        antialias_order=int(_require(resampling, "antialias_order", f"{context}.resampling")),
        min_frequency_hz=float(_require(grid, "min_frequency_hz", f"{context}.grid")),
        max_frequency_hz=float(_require(grid, "max_frequency_hz", f"{context}.grid")),
        n_frequencies=int(_require(grid, "n_frequencies", f"{context}.grid")),
        n_times=int(_require(grid, "n_times", f"{context}.grid")),
        stft_window_cycles=float(_require(stft, "window_cycles", f"{context}.stft")),
        stft_window_type=str(_require(stft, "window_type", f"{context}.stft")),
        morlet_w0=float(_require(cwt, "morlet_w0", f"{context}.cwt")),
        reference_power=float(_require(decibel, "reference_power", f"{context}.decibel")),
        floor_power=float(_require(decibel, "floor_power", f"{context}.decibel")),
        window_search=_build_window_search_config(_require(section, "window_search", context)),
        anchoring=_build_anchoring_config(_require(section, "anchoring", context)),
        n_distribution_records=int(_require(section, "n_distribution_records", context)),
        distribution_plot_margin_db=float(_require(section, "distribution_plot_margin_db", context)),
        n_png_records_per_class=int(_require(section, "n_png_records_per_class", context)),
        output_dir=_resolve_path(_require(section, "output_dir", context), project_root),
        figures_dir=_resolve_path(_require(section, "figures_dir", context), project_root),
        tables_dir=_resolve_path(_require(section, "tables_dir", context), project_root),
    )
    nyquist_hz = config.target_sampling_rate_hz / 2.0
    problems = []
    if not 0.0 < config.min_frequency_hz < config.max_frequency_hz:
        problems.append("grid requires 0 < min_frequency_hz < max_frequency_hz")
    if config.max_frequency_hz > config.antialias_corner_hz:
        problems.append("max_frequency_hz must not exceed antialias_corner_hz")
    if config.antialias_corner_hz >= nyquist_hz:
        problems.append("antialias_corner_hz must be below the target Nyquist frequency")
    if config.anchor_method not in VALID_ANCHOR_METHODS:
        problems.append(f"window.anchor must be one of {VALID_ANCHOR_METHODS}")
    if config.pre_anchor_s < 0 or config.post_anchor_s <= 0:
        problems.append("window requires pre_anchor_s >= 0 and post_anchor_s > 0")
    if config.context_margin_s < max(config.stft_window_s / 2.0, 2.0 * config.morlet_half_width_s):
        problems.append(
            "context_margin_s must be at least half the STFT window "
            f"({config.stft_window_s / 2.0:.1f} s) and twice the Morlet half-width "
            f"({2.0 * config.morlet_half_width_s:.1f} s)"
        )
    if not 0.0 <= config.taper_fraction < 0.5:
        problems.append("taper_fraction must be in [0, 0.5)")
    if config.n_frequencies < 2 or config.n_times < 2:
        problems.append("grid sizes must be at least 2")
    if config.reference_power <= 0 or config.floor_power <= 0:
        problems.append("reference_power and floor_power must be positive")
    if problems:
        raise ValueError("Invalid preprocessing configuration: " + "; ".join(problems) + ".")
    return config


def _build_split_config(section: Mapping[str, Any], project_root: Path) -> SplitConfig:
    """Build and validate the ``split`` section.

    Args:
        section: Parsed ``split`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated SplitConfig.
    """
    config = SplitConfig(
        validation_fraction=float(_require(section, "validation_fraction", "split")),
        random_seed=int(_require(section, "random_seed", "split")),
        output_file=_resolve_path(_require(section, "output_file", "split"), project_root),
        tables_dir=_resolve_path(_require(section, "tables_dir", "split"), project_root),
    )
    if not 0.0 < config.validation_fraction <= 0.5:
        raise ValueError("split.validation_fraction must be in (0, 0.5].")
    return config


def _as_int_tuple(values: Any, context: str) -> tuple[int, ...]:
    """Convert a YAML list of integers into a tuple, validating its type.

    Args:
        values: Parsed YAML value.
        context: Entry name used in the error message.

    Returns:
        Tuple of integers.
    """
    if not isinstance(values, list) or not values or not all(isinstance(value, int) for value in values):
        raise TypeError(f"Configuration entry '{context}' must be a non-empty list of integers.")
    return tuple(values)


def _build_model_config(section: Mapping[str, Any], project_root: Path) -> ModelConfig:
    """Build and validate the ``model`` section.

    Args:
        section: Parsed ``model`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated ModelConfig.
    """
    context = "model"
    dropout = _require(section, "dropout", context)
    grad_cam = _require(section, "grad_cam", context)
    config = ModelConfig(
        conv_filters=_as_int_tuple(_require(section, "conv_filters", context), f"{context}.conv_filters"),
        conv_kernels=_as_int_tuple(_require(section, "conv_kernels", context), f"{context}.conv_kernels"),
        recurrent_units=int(_require(section, "recurrent_units", context)),
        recurrent_layers=int(_require(section, "recurrent_layers", context)),
        head_units=int(_require(section, "head_units", context)),
        conv_dropout=float(_require(dropout, "conv", f"{context}.dropout")),
        recurrent_dropout=float(_require(dropout, "recurrent", f"{context}.dropout")),
        head_dropout=float(_require(dropout, "head", f"{context}.dropout")),
        mc_samples=int(_require(section, "mc_samples", context)),
        grad_cam_mc_samples=int(_require(grad_cam, "mc_samples", f"{context}.grad_cam")),
        grad_cam_records_per_class=int(_require(grad_cam, "records_per_class", f"{context}.grad_cam")),
        figures_dir=_resolve_path(_require(section, "figures_dir", context), project_root),
        tables_dir=_resolve_path(_require(section, "tables_dir", context), project_root),
    )
    problems = []
    if len(config.conv_filters) != len(config.conv_kernels):
        problems.append("conv_filters and conv_kernels must have the same length")
    if any(kernel < 3 or kernel % 2 == 0 for kernel in config.conv_kernels):
        problems.append("conv_kernels must be odd and at least 3")
    if min(config.conv_filters) < 1 or config.recurrent_units < 1 or config.head_units < 1:
        problems.append("filters and units must be positive")
    if config.recurrent_layers < 1:
        problems.append("recurrent_layers must be at least 1")
    if not all(0.0 <= rate < 1.0 for rate in (config.conv_dropout, config.recurrent_dropout, config.head_dropout)):
        problems.append("dropout rates must be in [0, 1)")
    if config.mc_samples < 1 or config.grad_cam_mc_samples < 2:
        problems.append("mc_samples must be at least 1 and grad_cam.mc_samples at least 2")
    if config.grad_cam_records_per_class < 1:
        problems.append("grad_cam.records_per_class must be at least 1")
    if problems:
        raise ValueError("Invalid model configuration: " + "; ".join(problems) + ".")
    return config


def _build_training_config(section: Mapping[str, Any], project_root: Path) -> TrainingConfig:
    """Build and validate the ``training`` section.

    Args:
        section: Parsed ``training`` mapping.
        project_root: Base directory for relative paths.

    Returns:
        Validated TrainingConfig.
    """
    context = "training"
    config = TrainingConfig(
        representation=str(_require(section, "representation", context)),
        feature_source=str(_require(section, "feature_source", context)),
        batch_size=int(_require(section, "batch_size", context)),
        learning_rate=float(_require(section, "learning_rate", context)),
        weight_decay=float(_require(section, "weight_decay", context)),
        max_epochs=int(_require(section, "max_epochs", context)),
        lr_factor=float(_require(section, "lr_factor", context)),
        lr_patience=int(_require(section, "lr_patience", context)),
        early_stopping_patience=int(_require(section, "early_stopping_patience", context)),
        num_workers=int(_require(section, "num_workers", context)),
        device=str(_require(section, "device", context)),
        random_seed=int(_require(section, "random_seed", context)),
        calibration_bins=int(_require(section, "calibration_bins", context)),
        runs_dir=_resolve_path(_require(section, "runs_dir", context), project_root),
    )
    problems = []
    if config.representation not in ("spectrogram", "scalogram"):
        problems.append("representation must be spectrogram or scalogram")
    if config.feature_source not in VALID_WAVEFORM_SOURCES:
        problems.append(f"feature_source must be one of {VALID_WAVEFORM_SOURCES}")
    if config.device not in ("auto", "cpu", "mps", "cuda"):
        problems.append("device must be auto, cpu, mps or cuda")
    if min(config.batch_size, config.max_epochs, config.calibration_bins) < 1:
        problems.append("batch_size, max_epochs and calibration_bins must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0 or not 0.0 < config.lr_factor < 1.0:
        problems.append("learning_rate must be positive, weight_decay non-negative, lr_factor in (0, 1)")
    if min(config.lr_patience, config.early_stopping_patience, config.num_workers) < 0:
        problems.append("patiences and num_workers must be non-negative")
    if problems:
        raise ValueError("Invalid training configuration: " + "; ".join(problems) + ".")
    return config


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
        preprocessing=_build_preprocessing_config(_require(raw, "preprocessing", "root"), root),
        split=_build_split_config(_require(raw, "split", "root"), root),
        model=_build_model_config(_require(raw, "model", "root"), root),
        training=_build_training_config(_require(raw, "training", "root"), root),
        output=_build_output_config(_require(raw, "output", "root"), root),
    )


def ensure_output_dirs(output: OutputConfig) -> None:
    """Create the output directories if they do not exist.

    Args:
        output: Output configuration.
    """
    output.figures_dir.mkdir(parents=True, exist_ok=True)
    output.tables_dir.mkdir(parents=True, exist_ok=True)
