"""Feature file of a representation: the main file (spectrogram, scalogram) or the obspy_spectrogram file.

The notebooks that train and evaluate a network locate and check its feature file through these
functions, so that every representation is read in the same way.
"""

from __future__ import annotations

from pathlib import Path

from wqu.config import OBSPY_REPRESENTATION, PreprocessingConfig
from wqu.preprocessing.obspy_features import check_obspy_features, obspy_feature_path
from wqu.preprocessing.pipeline import check_feature_placement, feature_file_path


def representation_feature_path(preprocessing: PreprocessingConfig, source: str, representation: str) -> Path:
    """Feature file holding a representation.

    Args:
        preprocessing: Preprocessing configuration.
        source: Waveform source name (e.g. ``seisbench``).
        representation: Representation (``training.representation``).

    Returns:
        Path of the ``obspy_spectrogram`` file for that representation, of the main feature file otherwise.
    """
    if representation == OBSPY_REPRESENTATION:
        return obspy_feature_path(preprocessing, source)
    return feature_file_path(preprocessing, source)


def check_representation_features(features_path: Path, preprocessing: PreprocessingConfig, representation: str) -> None:
    """Check that a feature file exists and matches the configuration.

    Args:
        features_path: Output of ``representation_feature_path``.
        preprocessing: Preprocessing configuration.
        representation: Representation (``training.representation``).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file was written with other settings.
    """
    if representation == OBSPY_REPRESENTATION:
        check_obspy_features(features_path, preprocessing)
    else:
        check_feature_placement(features_path, preprocessing)
