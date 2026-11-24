"""Classifier with Inception blocks, adapted from the InceptionNet variant of Shalabi et al. (2026).

Structure (scheme of Fig. 6 of the paper, with the settings of ``model.inception``):

- stem: 7 x 7 convolution with stride 2, 3 x 3 convolution with stride 2, 3 x 3 max pooling
  with stride 2 (each halves frequencies and times);
- Inception blocks (as block A of Inception v3, Szegedy et al., 2016) with four branches of a
  quarter of the output channels each: 1 x 1; 1 x 1 reduction then 5 x 5; 1 x 1 reduction then
  two 3 x 3; 3 x 3 average pooling then 1 x 1 (reductions to an eighth of the output channels);
  a 3 x 3 max pooling with stride 2 after the blocks of ``pool_after_blocks``;
- head: mean over frequencies and data time bins of the last block, dropout, output layer.

Differences from the code of the paper:

- every convolution is followed by batch normalization and ReLU (the code of the paper has no
  activation inside or after the Inception blocks, which then act as one linear map);
- channel dropout after every block, so that MC dropout samples the whole network (the paper
  has dropout only before the output layer);
- the output is a logit (no sigmoid), to be used with ``BCEWithLogitsLoss``;
- with time masking (``wqu.models.masking``) the padded time bins are set to zero before every
  convolution and pooling, the mask is halved at every stride-2 step, and the head averages
  over the data bins only.

Input: tensor of shape (batch, components, frequencies, times), followed by the time-mask
channel with time masking.
"""

from __future__ import annotations

import torch
from torch import nn

from wqu.config import ModelConfig
from wqu.models.masking import apply_time_mask, halve_time_mask, masked_mean, split_time_mask_channel

STEM_KERNELS: tuple[int, int] = (7, 3)
POOL_KERNEL = 3
DOWNSAMPLING_STRIDE = 2
BRANCH_SHARE = 4
REDUCTION_SHARE = 8


class ConvUnit(nn.Module):
    """Convolution (no bias), batch normalization and ReLU, with the padded time bins zeroed."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        """Create the unit.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Odd kernel size ("same" padding).
            stride: Stride of the convolution (1 or 2).
        """
        super().__init__()
        self.stride = stride
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the unit to masked inputs.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).
            mask: Boolean time mask of shape (batch, times), or None.

        Returns:
            Tuple (output with the padded bins at zero, mask of the output).
        """
        output_mask = halve_time_mask(mask) if self.stride == DOWNSAMPLING_STRIDE else mask
        return apply_time_mask(self.layers(apply_time_mask(inputs, mask)), output_mask), output_mask


class MaskedMaxPool(nn.Module):
    """3 x 3 max pooling with stride 2 (halves both axes), on non-negative masked activations."""

    def __init__(self) -> None:
        """Create the pooling layer."""
        super().__init__()
        self.pool = nn.MaxPool2d(POOL_KERNEL, stride=DOWNSAMPLING_STRIDE, padding=POOL_KERNEL // 2)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pool the inputs (padded bins at zero, below every ReLU output) and halve the mask.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times), non-negative.
            mask: Boolean time mask of shape (batch, times), or None.

        Returns:
            Tuple (pooled tensor with the padded bins at zero, halved mask).
        """
        output_mask = halve_time_mask(mask)
        return apply_time_mask(self.pool(apply_time_mask(inputs, mask)), output_mask), output_mask


class InceptionBlock(nn.Module):
    """Four parallel branches concatenated along the channels; the output keeps the input size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Create the block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels, a multiple of 8 (four branches of a quarter each).
        """
        super().__init__()
        branch = out_channels // BRANCH_SHARE
        reduction = out_channels // REDUCTION_SHARE
        self.branch1x1 = nn.ModuleList([ConvUnit(in_channels, branch, 1)])
        self.branch5x5 = nn.ModuleList([ConvUnit(in_channels, reduction, 1), ConvUnit(reduction, branch, 5)])
        self.branch3x3dbl = nn.ModuleList(
            [ConvUnit(in_channels, reduction, 1), ConvUnit(reduction, branch, 3), ConvUnit(branch, branch, 3)]
        )
        self.pool = nn.AvgPool2d(POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2)
        self.branch_pool = nn.ModuleList([ConvUnit(in_channels, branch, 1)])

    @staticmethod
    def _run_branch(units: nn.ModuleList, inputs: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Apply the units of one branch in order."""
        features = inputs
        for unit in units:
            features, _ = unit(features, mask)
        return features

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Concatenate the four branches.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).
            mask: Boolean time mask of shape (batch, times), or None.

        Returns:
            Tensor of shape (batch, out_channels, frequencies, times), padded bins at zero.
        """
        masked = apply_time_mask(inputs, mask)
        pooled = apply_time_mask(self.pool(masked), mask)
        return torch.cat(
            [
                self._run_branch(self.branch1x1, masked, mask),
                self._run_branch(self.branch5x5, masked, mask),
                self._run_branch(self.branch3x3dbl, masked, mask),
                self._run_branch(self.branch_pool, pooled, mask),
            ],
            dim=1,
        )


class InceptionClassifier(nn.Module):
    """Record-level quality classifier: stem, Inception blocks, mean over the data bins, output layer."""

    def __init__(self, config: ModelConfig, n_components: int) -> None:
        """Create the network.

        Args:
            config: Model configuration with the ``inception`` settings.
            n_components: Input channels (waveform components).

        Raises:
            ValueError: If the configuration has no Inception settings.
        """
        super().__init__()
        if config.inception is None:
            raise ValueError("The Inception network needs the model.inception settings.")
        settings = config.inception
        self.time_masking = config.time_masking
        self.n_components = n_components
        self.stem = nn.ModuleList(
            [
                ConvUnit(n_components, settings.stem_filters, STEM_KERNELS[0], stride=DOWNSAMPLING_STRIDE),
                ConvUnit(settings.stem_filters, settings.stem_filters, STEM_KERNELS[1], stride=DOWNSAMPLING_STRIDE),
                MaskedMaxPool(),
            ]
        )
        channels = (settings.stem_filters, *settings.block_filters)
        self.blocks = nn.ModuleList(
            InceptionBlock(channels[index], channels[index + 1]) for index in range(len(settings.block_filters))
        )
        self.pools = nn.ModuleDict(
            {str(block): MaskedMaxPool() for block in sorted(set(settings.pool_after_blocks))}
        )
        self.block_dropout = nn.Dropout2d(settings.block_dropout)
        self.head = nn.Sequential(nn.Dropout(settings.head_dropout), nn.Linear(settings.block_filters[-1], 1))

    @property
    def last_conv_layer(self) -> nn.Module:
        """Last Inception block (before its dropout), used by Grad-CAM."""
        return self.blocks[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (batch, channels, frequencies, times) to logits of shape (batch,).

        The channels are the components, followed by the time mask when ``time_masking`` is set.
        """
        features, mask = split_time_mask_channel(inputs, self.n_components, self.time_masking)
        for layer in self.stem:
            features, mask = layer(features, mask)
        for number, block in enumerate(self.blocks, start=1):
            features = self.block_dropout(block(features, mask))
            if str(number) in self.pools:
                features, mask = self.pools[str(number)](features, mask)
        return self.head(masked_mean(features, mask, (2, 3))).squeeze(-1)
