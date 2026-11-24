"""Feature file of the ``obspy_spectrogram`` images (``wqu.preprocessing.obspy_spectrogram``).

The file has the layout of the main feature file, so that the split, the training and the
evaluation read it in the same way: one dataset named after the representation, of shape
(records, components, n_frequencies, n_times) in float16, the row and column centres
(``frequencies_hz``, ``times_s``; see ``image_axes`` for their units) and the settings as
attributes. Row ``i`` is the record with ``feature_index == i`` in ``records.csv``.

Records without any component with data are kept in the file (as images of zeros) but left out of
``records.csv`` and listed in ``excluded_records.csv``: they are the records that the time-masked
networks exclude, so the runs on this file use the same records as the other runs.

Each combination of settings has its own folder, so variants of the images coexist. The file is
written under a temporary name with a flag per record and renamed at the end; an interrupted run
resumes from the records not yet flagged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from wqu.config import OBSPY_REPRESENTATION, WINDOW_TIME_AXIS, PreprocessingConfig
from wqu.data.loading import CV_COMPONENTS
from wqu.preprocessing.obspy_spectrogram import image_axes, record_image
from wqu.preprocessing.pipeline import (
    FEATURE_DTYPE,
    FEATURE_INDEX_COLUMN,
    PROCESSED_DATASET,
    RECORD_KEYS_DATASET,
    WINDOW_PLACEMENT_ATTRIBUTE,
    feature_records_path,
)

OBSPY_FOLDER = "obspy"
EXCLUDED_RECORDS_FILE = "excluded_records.csv"
SAMPLING_RATE_DATASET = "sampling_rate_hz"
DURATION_DATASET = "duration_s"
HAS_DATA_DATASET = "has_data"
SAMPLING_RATE_COLUMN = "obspy_sampling_rate_hz"
DURATION_COLUMN_FORMAT = "obspy_duration_s_{}"
HAS_DATA_COLUMN_FORMAT = "obspy_has_data_{}"
SETTING_ATTRIBUTE_FORMAT = "obspy_{}"


def obspy_feature_path(preprocessing: PreprocessingConfig, source: str) -> Path:
    """Path of the feature file of the configured ``obspy_spectrogram`` images.

    Args:
        preprocessing: Preprocessing configuration with the ``obspy_spectrogram`` settings.
        source: Waveform source name (e.g. ``seisbench``).

    Returns:
        ``output_dir/<source>/obspy/<variant>/features.h5``, with the window placement as a further
        subfolder for the ``window`` time axis.

    Raises:
        ValueError: If the configuration has no ``obspy_spectrogram`` settings.
    """
    settings = preprocessing.obspy_spectrogram
    if settings is None:
        raise ValueError("The configuration has no preprocessing.obspy_spectrogram settings.")
    folder = preprocessing.output_dir / source / OBSPY_FOLDER / settings.variant_name
    if settings.time_axis == WINDOW_TIME_AXIS:
        folder = folder / preprocessing.window_placement
    return folder / "features.h5"


def excluded_records_path(features_path: Path) -> Path:
    """Path of the table of the records without data, next to an ``obspy_spectrogram`` feature file.

    Args:
        features_path: Feature file.

    Returns:
        Path of ``excluded_records.csv``.
    """
    return features_path.with_name(EXCLUDED_RECORDS_FILE)


def setting_attributes(preprocessing: PreprocessingConfig) -> dict[str, object]:
    """Attributes that identify the images of a feature file.

    Args:
        preprocessing: Preprocessing configuration with the ``obspy_spectrogram`` settings.

    Returns:
        Mapping from attribute name to value (image settings, decibel floor, window placement).
    """
    attributes: dict[str, object] = {
        SETTING_ATTRIBUTE_FORMAT.format(key): value for key, value in asdict(preprocessing.obspy_spectrogram).items()
    }
    attributes["floor_power"] = preprocessing.floor_power
    attributes[WINDOW_PLACEMENT_ATTRIBUTE] = preprocessing.window_placement
    return attributes


def check_obspy_features(features_path: Path, preprocessing: PreprocessingConfig) -> None:
    """Check that a feature file holds the images of the configured settings.

    Args:
        features_path: Feature file written by ``store_obspy_features``.
        preprocessing: Preprocessing configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a stored setting differs from the configuration.
    """
    if not features_path.is_file():
        raise FileNotFoundError(f"{features_path} does not exist: run notebook 02d with the current settings.")
    expected = setting_attributes(preprocessing)
    if preprocessing.obspy_spectrogram.time_axis != WINDOW_TIME_AXIS:
        expected.pop(WINDOW_PLACEMENT_ATTRIBUTE)
    with h5py.File(features_path, mode="r") as stored:
        different = {key: (stored.attrs.get(key), value) for key, value in expected.items()
                     if stored.attrs.get(key) != value}
    if different:
        raise ValueError(f"{features_path} was written with other settings (stored, configured): {different}.")


def _open_obspy_file(path: Path, records: pd.DataFrame, preprocessing: PreprocessingConfig) -> h5py.File:
    """Open an unfinished feature file to resume it, or create it.

    Args:
        path: Temporary HDF5 file.
        records: Records to process, in storage order.
        preprocessing: Preprocessing configuration (settings stored as attributes of a new file).

    Returns:
        File open for writing.

    Raises:
        ValueError: If an existing unfinished file was created for other records or settings.
    """
    settings = preprocessing.obspy_spectrogram
    keys = records.index.astype(str).to_numpy(dtype=object)
    shape = (len(records), len(CV_COMPONENTS), settings.n_frequencies, settings.n_times)
    attributes = setting_attributes(preprocessing)
    if path.exists():
        output = h5py.File(path, mode="r+")
        stored_keys = output[RECORD_KEYS_DATASET].asstr()[()]
        same_settings = all(output.attrs.get(key) == value for key, value in attributes.items())
        if len(stored_keys) != len(keys) or (stored_keys != keys).any() or not same_settings:
            output.close()
            raise ValueError(f"{path} was created for other records or settings; delete it to start again.")
        return output
    path.parent.mkdir(parents=True, exist_ok=True)
    output = h5py.File(path, mode="w")
    output.create_dataset(OBSPY_REPRESENTATION, shape=shape, dtype=FEATURE_DTYPE, chunks=(1, *shape[1:]))
    rows, columns = image_axes(settings)
    output.create_dataset("frequencies_hz", data=rows)
    output.create_dataset("times_s", data=columns)
    output.create_dataset(RECORD_KEYS_DATASET, data=keys, dtype=h5py.string_dtype())
    output.create_dataset(PROCESSED_DATASET, data=np.zeros(len(records), dtype=bool))
    output.create_dataset(SAMPLING_RATE_DATASET, data=np.full(len(records), np.nan))
    output.create_dataset(DURATION_DATASET, data=np.zeros((len(records), len(CV_COMPONENTS))))
    output.create_dataset(HAS_DATA_DATASET, data=np.zeros((len(records), len(CV_COMPONENTS)), dtype=bool))
    for key, value in attributes.items():
        output.attrs[key] = value
    return output


def _image_table(output: h5py.File, records: pd.DataFrame) -> pd.DataFrame:
    """Records with their feature index, sampling rate, durations and data flags, read from the file."""
    table = records.copy()
    table[FEATURE_INDEX_COLUMN] = np.arange(len(records))
    table[SAMPLING_RATE_COLUMN] = output[SAMPLING_RATE_DATASET][()]
    durations = output[DURATION_DATASET][()]
    has_data = output[HAS_DATA_DATASET][()]
    for position, component in enumerate(CV_COMPONENTS):
        table[DURATION_COLUMN_FORMAT.format(component)] = durations[:, position]
        table[HAS_DATA_COLUMN_FORMAT.format(component)] = has_data[:, position]
    return table


def store_obspy_features(
    records: pd.DataFrame,
    batch_source: Callable[[pd.DataFrame], Iterable[tuple[pd.DataFrame, np.ndarray]]],
    stored_rate_hz: float,
    preprocessing: PreprocessingConfig,
    tolerance: float,
    output_path: Path,
) -> pd.DataFrame:
    """Compute the image of every record and store it; resumable.

    Args:
        records: Records to process (their index is used to match the batches).
        batch_source: Function returning an iterable of (records, traces) for a subset of ``records``.
        stored_rate_hz: Sampling rate of the stored traces.
        preprocessing: Preprocessing configuration with the ``obspy_spectrogram`` settings.
        tolerance: Zero threshold used to find the zero padding.
        output_path: HDF5 file to create.

    Returns:
        Table of the records with at least one component with data, with ``feature_index``, sampling
        rate, segment durations and data flags; also written as ``records.csv`` next to the file,
        while the other records are written to ``excluded_records.csv``.

    Raises:
        RuntimeError: If a record cannot be processed (the message names the record) or some
            records are not returned by the batch source.
    """
    positions = pd.Series(np.arange(len(records)), index=records.index)
    temporary_path = output_path.with_suffix(".partial.h5")
    with _open_obspy_file(temporary_path, records, preprocessing) as output:
        processed = output[PROCESSED_DATASET][()]
        with tqdm(total=len(records), initial=int(processed.sum()), desc="obspy spectrograms") as progress:
            for batch, traces in batch_source(records[~processed]):
                for (index, record), trace in zip(batch.iterrows(), traces):
                    try:
                        result = record_image(trace, record, stored_rate_hz, preprocessing, tolerance)
                    except (ValueError, IndexError, ZeroDivisionError) as error:
                        raise RuntimeError(f"Record {index} could not be processed: {error}") from error
                    row = int(positions[index])
                    output[OBSPY_REPRESENTATION][row] = result.image.astype(FEATURE_DTYPE)
                    output[SAMPLING_RATE_DATASET][row] = result.sampling_rate_hz
                    output[DURATION_DATASET][row] = result.durations_s
                    output[HAS_DATA_DATASET][row] = result.has_data
                    output[PROCESSED_DATASET][row] = True
                    progress.update(1)
                output.flush()
        if not output[PROCESSED_DATASET][()].all():
            raise RuntimeError("Some records were not returned by the batch source; the file is left unfinished.")
        table = _image_table(output, records)
    temporary_path.replace(output_path)
    with_data = table[[HAS_DATA_COLUMN_FORMAT.format(component) for component in CV_COMPONENTS]].any(axis=1)
    table[with_data].to_csv(feature_records_path(output_path))
    table[~with_data].to_csv(excluded_records_path(output_path))
    return table[with_data]
