"""Export of the stored ObsPy-style images as PNG files read by the code of Shalabi et al. (2026).

The published code of Shalabi et al. reads one 8-bit grey-level PNG per component, named
``<class prefix>_<channel code>_<index>.png`` in one folder per class (``GoodQuality/GQ_HNE_12.png``,
``BadQuality/BQ_HNZ_7.png``), and stacks the three components of each index into one input. Exporting
the images of notebook 02d in this layout lets their training code run on our data unchanged.

The stored grey levels (values in [0, 1]) are written as 8-bit integers (round(255 x value)), the
quantization of the PNG figures of Shalabi et al., with the highest frequency in the top row as in a
spectrogram figure. A table maps every exported index to the record it comes from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

from wqu.config import OBSPY_REPRESENTATION
from wqu.preprocessing.pipeline import FEATURE_INDEX_COLUMN

GREY_LEVELS = 255
FILE_NAME_FORMAT = "{prefix}_{channel}_{index}.png"
EXPORT_TABLE_FILE = "exported_records.csv"
EXPORT_INDEX_COLUMN = "export_index"


@dataclass(frozen=True)
class ClassFolder:
    """Folder and file prefix of the PNG images of one class.

    Attributes:
        folder: Name of the folder of the class (e.g. ``GoodQuality``).
        prefix: Prefix of the file names (e.g. ``GQ``).
    """

    folder: str
    prefix: str


def grey_levels_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert grey levels in [0, 1] to 8-bit integers.

    Args:
        image: Array of grey levels in [0, 1].

    Returns:
        Array of the same shape with dtype uint8 (round(255 x value)).

    Raises:
        ValueError: If a value is outside [0, 1] or not finite.
    """
    values = np.asarray(image, dtype=np.float64)
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("Grey levels must be finite and within [0, 1].")
    return np.rint(values * GREY_LEVELS).astype(np.uint8)


def figure_orientation(component_image: np.ndarray) -> np.ndarray:
    """Component image with the highest frequency in the top row, as in a spectrogram figure.

    Args:
        component_image: Image of shape (frequencies, times), lowest frequency in row 0.

    Returns:
        Image flipped along the frequency axis.
    """
    return component_image[::-1, :]


def png_path(output_dir: Path, class_folder: ClassFolder, channel: str, index: int) -> Path:
    """Path of the PNG of one component of one exported record.

    Args:
        output_dir: Folder holding one subfolder per class.
        class_folder: Folder and prefix of the class.
        channel: Channel code of the component (e.g. ``HNE``).
        index: Index of the exported record.

    Returns:
        ``output_dir/<folder>/<prefix>_<channel>_<index>.png``.
    """
    return output_dir / class_folder.folder / FILE_NAME_FORMAT.format(prefix=class_folder.prefix, channel=channel,
                                                                      index=index)


def write_record_pngs(image: np.ndarray, output_dir: Path, class_folder: ClassFolder, channels: Sequence[str],
                      index: int) -> None:
    """Write the PNG of every component of one record.

    Args:
        image: Stored image of shape (components, frequencies, times), grey levels in [0, 1].
        output_dir: Folder holding one subfolder per class.
        class_folder: Folder and prefix of the class of the record.
        channels: Channel code of each component, in the order of the stored components.
        index: Index of the exported record.

    Raises:
        ValueError: If the number of channel codes differs from the number of components.
    """
    if len(channels) != image.shape[0]:
        raise ValueError(f"{len(channels)} channel codes for {image.shape[0]} components.")
    for component_image, channel in zip(image, channels):
        pixels = grey_levels_to_uint8(figure_orientation(component_image))
        Image.fromarray(pixels).save(png_path(output_dir, class_folder, channel, index))


def export_png_images(
    features_path: Path,
    records: pd.DataFrame,
    label_column: str,
    class_folders: Mapping[int, ClassFolder],
    channels: Sequence[str],
    output_dir: Path,
    batch_size: int,
) -> pd.DataFrame:
    """Write the images of the records as PNG files in the layout of Shalabi et al.

    The export index of a record is its feature index, unique across the classes.

    Args:
        features_path: Feature file of notebook 02d.
        records: Records to export, with the feature index and the label.
        label_column: Label column.
        class_folders: Folder and prefix of each label.
        channels: Channel code of each stored component, in storage order.
        output_dir: Folder that receives one subfolder per class and the table of the exported records.
        batch_size: Records read from the feature file at a time.

    Returns:
        Table of the exported records (original index, export index, label, class folder), also written
        to ``output_dir/exported_records.csv``.

    Raises:
        KeyError: If a label has no class folder.
    """
    missing = sorted(set(records[label_column].unique()) - set(class_folders))
    if missing:
        raise KeyError(f"No class folder for the labels {missing}.")
    for class_folder in class_folders.values():
        (output_dir / class_folder.folder).mkdir(parents=True, exist_ok=True)
    ordered = records.sort_values(FEATURE_INDEX_COLUMN)
    feature_rows = ordered[FEATURE_INDEX_COLUMN].to_numpy()
    labels = ordered[label_column].to_numpy()
    with h5py.File(features_path, mode="r") as stored:
        images = stored[OBSPY_REPRESENTATION]
        for start in range(0, len(ordered), batch_size):
            rows = feature_rows[start:start + batch_size]
            batch = images[rows[0]:rows[-1] + 1]
            for row, label in zip(rows, labels[start:start + batch_size]):
                write_record_pngs(batch[row - rows[0]], output_dir, class_folders[int(label)], channels, int(row))
    table = pd.DataFrame({
        EXPORT_INDEX_COLUMN: feature_rows,
        label_column: labels,
        "class_folder": [class_folders[int(label)].folder for label in labels],
    }, index=ordered.index)
    table.to_csv(output_dir / EXPORT_TABLE_FILE)
    return table
