"""Reading of the stored features for training, with per-frequency standardization.

The feature file holds decibel images of shape (records, components, frequencies,
times). Each frequency is standardized with a mean and a standard deviation common
to the three components and to all times, computed on the training records only;
the stored values stay in decibels, so the normalization can change without
recomputing the features.

With time masking, each image gets one more channel holding the time mask of the
record (1 on the time bins with recorded data, 0 on the zero padding), repeated at
every frequency; the network separates it from the components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

STATISTICS_BLOCK_SIZE = 256


@dataclass(frozen=True)
class FrequencyStatistics:
    """Per-frequency mean and standard deviation of the training images.

    Attributes:
        mean: Shape (frequencies,).
        std: Shape (frequencies,).
    """

    mean: np.ndarray
    std: np.ndarray

    def save(self, path: Path) -> Path:
        """Write the statistics to an ``.npz`` file.

        Args:
            path: Destination file.

        Returns:
            The destination path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        return path

    @classmethod
    def load(cls, path: Path) -> FrequencyStatistics:
        """Read statistics written by ``save``.

        Args:
            path: ``.npz`` file.

        Returns:
            FrequencyStatistics.
        """
        with np.load(path) as stored:
            return cls(mean=stored["mean"], std=stored["std"])


def frequency_statistics(path: Path, representation: str, rows: np.ndarray) -> FrequencyStatistics:
    """Per-frequency mean and standard deviation over the given records.

    The records are read in sorted blocks and accumulated in double precision, so
    the whole set never needs to fit in memory.

    Args:
        path: Feature file.
        representation: ``spectrogram`` or ``scalogram``.
        rows: Feature indices of the training records.

    Returns:
        FrequencyStatistics.

    Raises:
        ValueError: If no rows are given or a frequency is constant.
    """
    if len(rows) == 0:
        raise ValueError("No rows to compute the statistics on.")
    sorted_rows = np.sort(np.asarray(rows))
    total = total_squares = None
    count = 0
    with h5py.File(path, mode="r") as stored:
        dataset = stored[representation]
        for start in range(0, len(sorted_rows), STATISTICS_BLOCK_SIZE):
            block = dataset[sorted_rows[start : start + STATISTICS_BLOCK_SIZE]].astype(np.float64)
            block_sum = block.sum(axis=(0, 1, 3))
            block_squares = (block**2).sum(axis=(0, 1, 3))
            total = block_sum if total is None else total + block_sum
            total_squares = block_squares if total_squares is None else total_squares + block_squares
            count += block.shape[0] * block.shape[1] * block.shape[3]
    mean = total / count
    std = np.sqrt(np.maximum(total_squares / count - mean**2, 0.0))
    if np.any(std == 0):
        raise ValueError("At least one frequency is constant over the training records.")
    return FrequencyStatistics(mean=mean.astype(np.float32), std=std.astype(np.float32))


def with_time_mask_channel(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Append the time mask to images as one more channel, constant over frequency.

    Args:
        images: Images of shape (..., channels, frequencies, times).
        masks: Time masks of shape (..., times), true where the time bin has recorded data.

    Returns:
        Float32 array of shape (..., channels + 1, frequencies, times).
    """
    mask_channel = np.broadcast_to(np.asarray(masks, dtype=np.float32)[..., None, None, :],
                                   (*images.shape[:-3], 1, *images.shape[-2:]))
    return np.concatenate([np.asarray(images, dtype=np.float32), mask_channel], axis=-3)


class FeatureDataset(Dataset):
    """Standardized images and labels read one record at a time from the feature file.

    The HDF5 file is opened lazily in each worker process.
    """

    def __init__(
        self,
        path: Path,
        representation: str,
        rows: np.ndarray,
        labels: np.ndarray,
        statistics: FrequencyStatistics,
        time_masks: np.ndarray | None = None,
    ) -> None:
        """Create the dataset.

        Args:
            path: Feature file.
            representation: ``spectrogram`` or ``scalogram``.
            rows: Feature indices of the records.
            labels: Labels of the records (0 = good, 1 = bad), aligned with ``rows``.
            statistics: Per-frequency statistics of the training records.
            time_masks: Time masks of all the records of the file, shape (records in file, times),
                indexed by feature index (``wqu.preprocessing.time_masks.read_time_masks``); None
                for images without the mask channel.
        """
        self.path = path
        self.representation = representation
        self.rows = np.asarray(rows)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.mean = statistics.mean[None, :, None]
        self.std = statistics.std[None, :, None]
        self.time_masks = time_masks
        self._file: h5py.File | None = None

    def __len__(self) -> int:
        """Number of records."""
        return len(self.rows)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized image of shape (channels, frequencies, times) and its label.

        The channels are the components, followed by the time mask when the dataset has masks.
        """
        if self._file is None:
            self._file = h5py.File(self.path, mode="r")
        row = int(self.rows[position])
        image = self._file[self.representation][row].astype(np.float32)
        standardized = (image - self.mean) / self.std
        if self.time_masks is not None:
            standardized = with_time_mask_channel(standardized, self.time_masks[row])
        return torch.from_numpy(np.ascontiguousarray(standardized)), torch.tensor(self.labels[position])


def make_loader(dataset: FeatureDataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    """Data loader with a seeded batch order.

    Args:
        dataset: Dataset.
        batch_size: Records per batch.
        shuffle: Whether to shuffle the records at every epoch.
        num_workers: Worker processes.
        seed: Seed of the batch order.

    Returns:
        DataLoader.
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, generator=generator)
