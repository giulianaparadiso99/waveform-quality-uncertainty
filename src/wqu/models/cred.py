"""Convolutional-recurrent classifier adapted from CRED (Mousavi et al., 2019).

Reimplemented in PyTorch from the description in the paper and the public Keras
code (https://github.com/smousavi05/CRED). Differences from CRED:

- one probability per record instead of one per time step: the LSTM outputs are
  averaged over time before a dense head;
- the final unidirectional LSTM (meant for real-time detection) is removed;
- dropout after every convolutional stage (channel dropout), every LSTM layer and
  in the head, so that MC dropout samples the whole network;
- the output is a logit (no sigmoid), to be used with ``BCEWithLogitsLoss``;
- L1 regularization of the dense layers is replaced by the optimizer weight decay.

Input: tensor of shape (batch, components, frequencies, times). Each convolutional
stage halves frequencies and times; the result is read as a sequence along time,
with channels and frequencies flattened into the features of each time step.
"""

from __future__ import annotations

import torch
from torch import nn

from wqu.config import ModelConfig


class ResidualBlock(nn.Module):
    """Two pre-activated convolutions (BN, ReLU, convolution) with an identity shortcut."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        """Create the block.

        Args:
            channels: Input and output channels.
            kernel_size: Odd kernel size of both convolutions.
        """
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return ``inputs + F(inputs)``."""
        return inputs + self.layers(inputs)


class ConvStage(nn.Module):
    """Strided convolution (halves both axes), residual block and channel dropout."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        """Create the stage.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Odd kernel size of the strided convolution; the residual block uses ``kernel_size - 2``.
            dropout: Channel dropout rate.
        """
        super().__init__()
        self.strided = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.residual = ResidualBlock(out_channels, kernel_size - 2)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the stage."""
        return self.dropout(self.residual(self.strided(inputs)))


class ResidualBiLSTM(nn.Module):
    """Stack of bidirectional LSTM layers with dropout; residual from the second layer on."""

    def __init__(self, input_size: int, units: int, n_layers: int, dropout: float) -> None:
        """Create the stack.

        Args:
            input_size: Features per time step of the input sequence.
            units: Hidden units per direction.
            n_layers: Number of layers.
            dropout: Dropout rate after each layer.
        """
        super().__init__()
        sizes = [input_size] + [2 * units] * (n_layers - 1)
        self.layers = nn.ModuleList(
            nn.LSTM(size, units, batch_first=True, bidirectional=True) for size in sizes
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Map (batch, steps, features) to (batch, steps, 2 * units)."""
        for position, layer in enumerate(self.layers):
            output = self.dropout(layer(sequence)[0])
            sequence = output if position == 0 else sequence + output
        return sequence


class CredClassifier(nn.Module):
    """Record-level quality classifier: convolutional stages, BiLSTM along time, dense head."""

    def __init__(self, config: ModelConfig, n_components: int, n_frequencies: int) -> None:
        """Create the network.

        Args:
            config: Model configuration.
            n_components: Input channels (waveform components).
            n_frequencies: Frequencies of the input grid.
        """
        super().__init__()
        channels = (n_components, *config.conv_filters)
        self.conv = nn.Sequential(
            *(
                ConvStage(channels[stage], channels[stage + 1], kernel, config.conv_dropout)
                for stage, kernel in enumerate(config.conv_kernels)
            )
        )
        reduced_frequencies = n_frequencies
        for _ in config.conv_kernels:
            reduced_frequencies = (reduced_frequencies + 1) // 2
        self.recurrent = ResidualBiLSTM(
            config.conv_filters[-1] * reduced_frequencies,
            config.recurrent_units,
            config.recurrent_layers,
            config.recurrent_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * config.recurrent_units, config.head_units),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_units, 1),
        )

    @property
    def last_conv_layer(self) -> nn.Module:
        """Residual block of the last convolutional stage (before its dropout), used by Grad-CAM."""
        return self.conv[-1].residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (batch, components, frequencies, times) to logits of shape (batch,)."""
        features = self.conv(inputs)
        batch, channels, frequencies, times = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(batch, times, channels * frequencies)
        summary = self.recurrent(sequence).mean(dim=1)
        return self.head(summary).squeeze(-1)


def build_model(config: ModelConfig, n_components: int, n_frequencies: int) -> CredClassifier:
    """Create the classifier from the configuration.

    Args:
        config: Model configuration.
        n_components: Input channels (waveform components).
        n_frequencies: Frequencies of the input grid.

    Returns:
        CredClassifier.
    """
    return CredClassifier(config, n_components, n_frequencies)
