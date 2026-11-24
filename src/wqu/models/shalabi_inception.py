"""InceptionNet variant of Shalabi et al. (2026), as in the code released with the paper.

Structure for 64 x 64 inputs (``InceptionNet_like_model`` of the paper code):

- stem: 7 x 7 convolution with 64 filters and stride 2, batch normalization, ReLU; 3 x 3
  convolution with 64 filters, batch normalization, ReLU; 3 x 3 max pooling with stride 2
  (64 x 64 -> 16 x 16);
- Inception blocks with 64 and 128 output channels, 3 x 3 max pooling with stride 2, blocks with
  256 and 512 output channels, 3 x 3 max pooling with stride 2 (16 x 16 -> 8 x 8 -> 4 x 4);
- each block has four branches of a quarter of the output channels: 1 x 1; 1 x 1 reduction to an
  eighth then 5 x 5; 1 x 1 reduction to an eighth then two 3 x 3; 3 x 3 average pooling then
  1 x 1; convolutions without bias, branches concatenated;
- head: average over frequencies and times, dropout, linear layer to one output.

Options (``model.shalabi_inception``):

- ``block_activation``: ``none`` = batch normalization of the concatenated branches and no
  activation, as in the paper code; ``after_concat`` = batch normalization and ReLU of the
  concatenated branches; ``every_conv`` = batch normalization and ReLU after every convolution of
  the branches (no further normalization of the concatenation);
- ``block_dropout``: channel dropout after every block, so that MC dropout samples the whole
  network (the paper has dropout only before the output layer; 0 reproduces it).

The output is a logit (the paper applies a sigmoid and a BCE loss: the same model, trained here
with ``BCEWithLogitsLoss``). The network has no time masking.
"""

from __future__ import annotations

import torch
from torch import nn

from wqu.config import (
    AFTER_CONCAT_BLOCK_ACTIVATION,
    EVERY_CONV_BLOCK_ACTIVATION,
    NO_BLOCK_ACTIVATION,
    VALID_BLOCK_ACTIVATIONS,
    ModelConfig,
)

STEM_FILTERS = 64
STEM_KERNELS: tuple[int, int] = (7, 3)
STEM_STRIDE = 2
BLOCK_FILTERS: tuple[int, ...] = (64, 128, 256, 512)
POOL_AFTER_BLOCKS: frozenset[int] = frozenset({2, 4})
POOL_KERNEL = 3
POOL_STRIDE = 2
BRANCH_SHARE = 4
REDUCTION_SHARE = 8


def branch_convolution(in_channels: int, out_channels: int, kernel_size: int, activated: bool) -> nn.Module:
    """Convolution of a branch ("same" padding, no bias), optionally with batch normalization and ReLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Odd kernel size.
        activated: Whether batch normalization and ReLU follow the convolution.

    Returns:
        The convolution, or a sequence of convolution, batch normalization and ReLU.
    """
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
    if not activated:
        return convolution
    return nn.Sequential(convolution, nn.BatchNorm2d(out_channels), nn.ReLU())


def stem_layers(n_components: int) -> nn.Sequential:
    """Stem of the network: two convolutions with batch normalization and ReLU, then max pooling.

    Args:
        n_components: Input channels.

    Returns:
        Sequential module reducing both axes by four.
    """
    first, second = STEM_KERNELS
    return nn.Sequential(
        nn.Conv2d(n_components, STEM_FILTERS, first, stride=STEM_STRIDE, padding=first // 2, bias=False),
        nn.BatchNorm2d(STEM_FILTERS),
        nn.ReLU(),
        nn.Conv2d(STEM_FILTERS, STEM_FILTERS, second, padding=second // 2, bias=False),
        nn.BatchNorm2d(STEM_FILTERS),
        nn.ReLU(),
        nn.MaxPool2d(POOL_KERNEL, stride=POOL_STRIDE, padding=POOL_KERNEL // 2),
    )


class ShalabiInceptionBlock(nn.Module):
    """Four parallel branches concatenated along the channels; the output keeps the input size."""

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        """Create the block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels, a multiple of 8.
            activation: ``none``, ``after_concat`` or ``every_conv`` (see the module docstring).

        Raises:
            ValueError: If the activation is unknown.
        """
        super().__init__()
        if activation not in VALID_BLOCK_ACTIVATIONS:
            raise ValueError(f"Unknown block activation '{activation}'.")
        every_conv = activation == EVERY_CONV_BLOCK_ACTIVATION
        branch = out_channels // BRANCH_SHARE
        reduction = out_channels // REDUCTION_SHARE
        self.branch1x1 = branch_convolution(in_channels, branch, 1, every_conv)
        self.branch5x5 = nn.Sequential(
            branch_convolution(in_channels, reduction, 1, every_conv),
            branch_convolution(reduction, branch, 5, every_conv),
        )
        self.branch3x3dbl = nn.Sequential(
            branch_convolution(in_channels, reduction, 1, every_conv),
            branch_convolution(reduction, branch, 3, every_conv),
            branch_convolution(branch, branch, 3, every_conv),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(POOL_KERNEL, stride=1, padding=POOL_KERNEL // 2),
            branch_convolution(in_channels, branch, 1, every_conv),
        )
        if activation == NO_BLOCK_ACTIVATION:
            self.output = nn.BatchNorm2d(out_channels)
        elif activation == AFTER_CONCAT_BLOCK_ACTIVATION:
            self.output = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU())
        else:
            self.output = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Concatenate the four branches and apply the output normalization.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).

        Returns:
            Tensor of shape (batch, out_channels, frequencies, times).
        """
        branches = [self.branch1x1(inputs), self.branch5x5(inputs), self.branch3x3dbl(inputs), self.branch_pool(inputs)]
        return self.output(torch.cat(branches, dim=1))


class ShalabiInceptionClassifier(nn.Module):
    """Record-level quality classifier: stem, Inception blocks with pooling, global average, output layer."""

    def __init__(self, config: ModelConfig, n_components: int) -> None:
        """Create the network.

        Args:
            config: Model configuration with the ``shalabi_inception`` settings.
            n_components: Input channels (waveform components).

        Raises:
            ValueError: If the settings are missing or time masking is requested.
        """
        super().__init__()
        if config.shalabi_inception is None:
            raise ValueError("The Shalabi Inception network needs the model.shalabi_inception settings.")
        if config.time_masking:
            raise ValueError("The Shalabi Inception network has no time masking.")
        settings = config.shalabi_inception
        self.stem = stem_layers(n_components)
        channels = (STEM_FILTERS, *BLOCK_FILTERS)
        self.blocks = nn.ModuleList(
            ShalabiInceptionBlock(channels[index], channels[index + 1], settings.block_activation)
            for index in range(len(BLOCK_FILTERS))
        )
        self.pool = nn.MaxPool2d(POOL_KERNEL, stride=POOL_STRIDE, padding=POOL_KERNEL // 2)
        self.block_dropout = nn.Dropout2d(settings.block_dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(settings.head_dropout),
            nn.Linear(BLOCK_FILTERS[-1], 1),
        )

    @property
    def last_conv_layer(self) -> nn.Module:
        """Last Inception block (before its dropout), used by Grad-CAM."""
        return self.blocks[-1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (batch, components, frequencies, times) to logits of shape (batch,)."""
        features = self.stem(inputs)
        for number, block in enumerate(self.blocks, start=1):
            features = self.block_dropout(block(features))
            if number in POOL_AFTER_BLOCKS:
                features = self.pool(features)
        return self.head(features).squeeze(-1)
