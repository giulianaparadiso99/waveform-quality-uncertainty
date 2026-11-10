"""Size and speed checks of a model before training."""

from __future__ import annotations

import time

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters.

    Args:
        model: Network.

    Returns:
        Number of parameters with ``requires_grad``.
    """
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def available_devices() -> list[str]:
    """Devices usable by PyTorch on this machine, CPU first.

    Returns:
        List among ``cpu``, ``mps`` (Apple GPU) and ``cuda``.
    """
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def synchronize(device: str) -> None:
    """Wait for the queued operations of an accelerator to finish.

    Args:
        device: Device name.
    """
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_training_step(
    model: nn.Module, loss: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, device: str, n_repeats: int
) -> float:
    """Average time of one training step (forward, backward, optimizer update).

    One warm-up step is run first and excluded. The model is moved to ``device``.

    Args:
        model: Network.
        loss: Loss module.
        inputs: Input batch.
        labels: Float labels of the batch.
        device: Device name.
        n_repeats: Timed steps.

    Returns:
        Seconds per step.
    """
    model.to(device).train()
    loss = loss.to(device)
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    def step() -> None:
        optimizer.zero_grad()
        loss(model(inputs), labels).backward()
        optimizer.step()

    step()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(n_repeats):
        step()
    synchronize(device)
    return (time.perf_counter() - start) / n_repeats
