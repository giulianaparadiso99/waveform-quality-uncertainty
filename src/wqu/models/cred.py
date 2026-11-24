"""Convolutional-recurrent classifier adapted from CRED (Mousavi et al., 2019).

Reimplemented in PyTorch from the description in the paper and the public Keras
code (https://github.com/smousavi05/CRED). Differences from CRED:

- one probability per record instead of one per time step: the LSTM outputs are
  averaged over time before a dense head;
- the final unidirectional LSTM (meant for real-time detection) is removed;
- dropout after every convolutional stage (channel dropout), every LSTM layer and
  in the head, so that MC dropout samples the whole network; optionally also between
  the two convolutions of each residual block (as in wide residual networks);
- the output is a logit (no sigmoid), to be used with ``BCEWithLogitsLoss``;
- L1 regularization of the dense layers is replaced by the optimizer weight decay.

Input: tensor of shape (batch, components, frequencies, times). Each convolutional
stage halves frequencies and times; the result is read as a sequence along time,
with channels and frequencies flattened into the features of each time step.

Time masking (``config.time_masking``): the input carries one more channel, the time
mask (1 where the time bin contains recorded data, 0 on the zero padding, the same at
every frequency). The network then never uses the padded bins:

- the input and the activations entering every convolution are set to zero on the padded
  bins, as the zero padding of the convolution itself does at the border of an image, so
  the values of the padded bins have no effect; the mask is halved with the time axis at
  every strided convolution (a bin is data if any of the two bins it summarizes is);
- the forward direction of each LSTM layer reads the data steps aligned to the start of
  the sequence and the backward direction reads them aligned to the end, so that neither
  direction passes through padded steps before the data;
- the mean over time uses the data steps only.

The data bins of a record must be contiguous (one recorded span) and at least one.
"""

from __future__ import annotations

import torch
from torch import nn

from wqu.config import INCEPTION_ARCHITECTURE, SHALABI_INCEPTION_ARCHITECTURE, ModelConfig
from wqu.models.inception import InceptionClassifier
from wqu.models.shalabi_inception import ShalabiInceptionClassifier

# The masking helpers live in wqu.models.masking; they are also importable from here, as before the split.
from wqu.models.masking import (  # noqa: F401
    MASK_THRESHOLD,
    apply_time_mask,
    data_span,
    halve_time_mask,
    masked_mean,
    shift_sequence,
    split_time_mask_channel,
)

class ResidualBlock(nn.Module):
    """Two pre-activated convolutions (BN, ReLU, convolution) with an identity shortcut.

    With a positive ``dropout``, channel dropout is applied before the second
    convolution (wide residual networks, Zagoruyko and Komodakis, 2016); with zero
    the block has no dropout layer at all, as in CRED.
    """

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.0) -> None:
        """Create the block.

        Args:
            channels: Input and output channels.
            kernel_size: Odd kernel size of both convolutions.
            dropout: Channel dropout rate between the two convolutions.
        """
        super().__init__()
        padding = kernel_size // 2
        inner_dropout = [nn.Dropout2d(dropout)] if dropout > 0 else []
        self.layers = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            *inner_dropout,
            nn.Conv2d(channels, channels, kernel_size, padding=padding),
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return ``inputs + F(inputs)``, with the padded bins set to zero before each convolution and at the end.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).
            mask: Boolean time mask of shape (batch, times), or None.

        Returns:
            Tensor of the same shape.
        """
        features = inputs
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                features = apply_time_mask(features, mask)
            features = layer(features)
        return apply_time_mask(inputs + features, mask)


class ConvStage(nn.Module):
    """Strided convolution (halves both axes), residual block and channel dropout."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dropout: float, residual_dropout: float = 0.0
    ) -> None:
        """Create the stage.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Odd kernel size of the strided convolution; the residual block uses ``kernel_size - 2``.
            dropout: Channel dropout rate after the stage.
            residual_dropout: Channel dropout rate inside the residual block.
        """
        super().__init__()
        self.strided = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2),
            nn.ReLU(),
        )
        self.residual = ResidualBlock(out_channels, kernel_size - 2, residual_dropout)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the stage.

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).
            mask: Boolean time mask of shape (batch, times), or None.

        Returns:
            Tuple (output with halved frequencies and times, mask of the output).
        """
        halved_mask = halve_time_mask(mask)
        features = apply_time_mask(self.strided(apply_time_mask(inputs, mask)), halved_mask)
        return self.dropout(self.residual(features, halved_mask)), halved_mask


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

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map (batch, steps, features) to (batch, steps, 2 * units).

        With a mask the sequences are first aligned so that their data steps start at step 0
        (the output is in these aligned coordinates, zero after the data); each layer then
        reads the data at the start of the sequence for the forward direction and at its
        end for the backward direction.

        Args:
            sequence: Tensor of shape (batch, steps, features).
            mask: Boolean mask of shape (batch, steps), or None.

        Returns:
            Tuple (output sequence, mask of the output: data steps at the start, or None).
        """
        if mask is None:
            for position, layer in enumerate(self.layers):
                output = self.dropout(layer(sequence)[0])
                sequence = output if position == 0 else sequence + output
            return sequence, None
        starts, lengths = data_span(mask)
        n_steps = sequence.shape[1]
        aligned_mask = torch.arange(n_steps, device=mask.device)[None, :] < lengths[:, None]
        to_end = lengths - n_steps
        sequence = shift_sequence(sequence, starts)
        for position, layer in enumerate(self.layers):
            forward_output = layer(sequence)[0][..., : layer.hidden_size]
            backward_output = layer(shift_sequence(sequence, to_end))[0][..., layer.hidden_size :]
            output = torch.cat([forward_output, shift_sequence(backward_output, -to_end)], dim=-1)
            output = self.dropout(output) * aligned_mask[:, :, None].to(output.dtype)
            sequence = output if position == 0 else sequence + output
        return sequence, aligned_mask


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
        self.time_masking = config.time_masking
        self.n_components = n_components
        channels = (n_components, *config.conv_filters)
        self.conv = nn.ModuleList(
            ConvStage(channels[stage], channels[stage + 1], kernel, config.conv_dropout, config.residual_dropout)
            for stage, kernel in enumerate(config.conv_kernels)
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

    def split_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Separate the images from the time-mask channel (only with time masking).

        Args:
            inputs: Tensor of shape (batch, channels, frequencies, times).

        Returns:
            Tuple (images, boolean mask of shape (batch, times) or None).

        Raises:
            ValueError: If the number of channels does not match the configuration.
        """
        return split_time_mask_channel(inputs, self.n_components, self.time_masking)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map (batch, channels, frequencies, times) to logits of shape (batch,).

        The channels are the components, followed by the time mask when ``time_masking`` is set.
        """
        features, mask = self.split_inputs(inputs)
        features = apply_time_mask(features, mask)
        for stage in self.conv:
            features, mask = stage(features, mask)
        batch, channels, frequencies, times = features.shape
        sequence = features.permute(0, 3, 1, 2).reshape(batch, times, channels * frequencies)
        sequence, mask = self.recurrent(sequence, mask)
        return self.head(masked_mean(sequence, mask, (1,))).squeeze(-1)


def build_model(config: ModelConfig, n_components: int, n_frequencies: int) -> nn.Module:
    """Create the classifier of the configured architecture (``model.architecture``).

    Args:
        config: Model configuration.
        n_components: Input channels (waveform components).
        n_frequencies: Frequencies of the input grid.

    Returns:
        CredClassifier, InceptionClassifier (``wqu.models.inception``) or ShalabiInceptionClassifier
        (``wqu.models.shalabi_inception``).
    """
    if config.architecture == INCEPTION_ARCHITECTURE:
        return InceptionClassifier(config, n_components)
    if config.architecture == SHALABI_INCEPTION_ARCHITECTURE:
        return ShalabiInceptionClassifier(config, n_components)
    return CredClassifier(config, n_components, n_frequencies)
