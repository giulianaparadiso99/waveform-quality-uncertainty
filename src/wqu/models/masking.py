"""Time masking shared by the networks: the zero-padded time bins are never used.

With time masking the input carries one more channel, the time mask (1 where the time bin
contains recorded data, 0 on the zero padding, the same at every frequency). The networks
set the activations of the padded bins to zero before every convolution and pooling, as the
zero padding of a convolution does at the border of an image, halve the mask together with the
time axis, and average over the data bins only.
"""

from __future__ import annotations

import torch
from torch import nn

MASK_THRESHOLD = 0.5


def apply_time_mask(features: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Set the padded time bins of a (batch, channels, frequencies, times) tensor to zero.

    Args:
        features: Activations or input images.
        mask: Boolean mask of shape (batch, times), or None (no masking).

    Returns:
        Masked tensor (the input itself when ``mask`` is None).
    """
    if mask is None:
        return features
    return features * mask[:, None, None, :].to(features.dtype)


def halve_time_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Mask after a stride-2 convolution: a bin is data if any of the two bins it covers is.

    Args:
        mask: Boolean mask of shape (batch, times), or None.

    Returns:
        Boolean mask of shape (batch, ceil(times / 2)), or None.
    """
    if mask is None:
        return None
    pooled = nn.functional.max_pool1d(mask[:, None, :].float(), kernel_size=2, stride=2, ceil_mode=True)
    return pooled[:, 0] > MASK_THRESHOLD


def data_span(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First data step and number of data steps of each sequence.

    Args:
        mask: Boolean mask of shape (batch, steps).

    Returns:
        Tuple (start, length), integer tensors of shape (batch,).

    Raises:
        ValueError: If a sequence has no data step or its data steps are not contiguous.
    """
    lengths = mask.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("A record has no time bin with recorded data: exclude it before the network.")
    starts = mask.float().argmax(dim=1)
    steps = torch.arange(mask.shape[1], device=mask.device)[None, :]
    expected = (steps >= starts[:, None]) & (steps < (starts + lengths)[:, None])
    if not bool(torch.equal(expected, mask)):
        raise ValueError("The time mask of a record is not a single contiguous span.")
    return starts, lengths


def shift_sequence(sequence: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Shift each sequence along time: output step ``t`` takes input step ``t + offset``.

    Steps falling outside the sequence are set to zero.

    Args:
        sequence: Tensor of shape (batch, steps, features).
        offsets: Integer tensor of shape (batch,).

    Returns:
        Shifted tensor of the same shape.
    """
    n_steps = sequence.shape[1]
    source = torch.arange(n_steps, device=sequence.device)[None, :] + offsets[:, None]
    inside = (source >= 0) & (source < n_steps)
    gathered = torch.gather(sequence, 1, source.clamp(0, n_steps - 1)[:, :, None].expand(-1, -1, sequence.shape[2]))
    return gathered * inside[:, :, None].to(sequence.dtype)


def split_time_mask_channel(
    inputs: torch.Tensor, n_components: int, time_masking: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Separate the images from the time-mask channel (only with time masking).

    Args:
        inputs: Tensor of shape (batch, channels, frequencies, times).
        n_components: Image channels (waveform components).
        time_masking: Whether the last channel is the time mask.

    Returns:
        Tuple (images, boolean mask of shape (batch, times) or None).

    Raises:
        ValueError: If the number of channels does not match.
    """
    expected = n_components + (1 if time_masking else 0)
    if inputs.shape[1] != expected:
        raise ValueError(
            f"Expected {expected} input channels ({n_components} components"
            f"{' and the time mask' if time_masking else ''}), got {inputs.shape[1]}."
        )
    if not time_masking:
        return inputs, None
    return inputs[:, :n_components], inputs[:, n_components, 0, :] > MASK_THRESHOLD


def masked_mean(features: torch.Tensor, mask: torch.Tensor | None, dims: tuple[int, ...]) -> torch.Tensor:
    """Mean of a tensor over some dimensions, counting only the data time bins.

    Args:
        features: Tensor of shape (batch, channels, frequencies, times) or (batch, times, features).
        mask: Boolean mask of shape (batch, times), or None (plain mean).
        dims: Dimensions averaged; with a mask they must include the time dimension.

    Returns:
        Tensor without the averaged dimensions.
    """
    if mask is None:
        return features.mean(dim=dims)
    if features.dim() == 4:
        weights = mask[:, None, None, :].to(features.dtype)
    else:
        weights = mask[:, :, None].to(features.dtype)
    weights = weights.expand_as(features)
    return (features * weights).sum(dim=dims) / weights.sum(dim=dims)
