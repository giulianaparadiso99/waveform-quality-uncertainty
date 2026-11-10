"""Monte Carlo dropout predictions.

At prediction time the dropout layers stay active while batch normalization uses
its running statistics, so that the stochastic passes differ only by the dropout
masks (Gal and Ghahramani, 2016).
"""

from __future__ import annotations

import torch
from torch import nn

DROPOUT_TYPES: tuple[type[nn.Module], ...] = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Put the model in evaluation mode with only the dropout layers active.

    Args:
        model: Network.

    Returns:
        The same model, modified in place.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()
    return model


@torch.no_grad()
def mc_dropout_probabilities(model: nn.Module, inputs: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Probabilities of the bad class from repeated stochastic passes.

    Args:
        model: Network returning logits of shape (batch,).
        inputs: Input batch.
        n_samples: Number of stochastic passes.

    Returns:
        Tensor of shape (n_samples, batch).
    """
    enable_mc_dropout(model)
    return torch.stack([torch.sigmoid(model(inputs)) for _ in range(n_samples)])
