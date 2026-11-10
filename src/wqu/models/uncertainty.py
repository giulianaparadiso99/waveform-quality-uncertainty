"""Monte Carlo dropout predictions.

At prediction time the dropout layers stay active while batch normalization uses
its running statistics, so that the stochastic passes differ only by the dropout
masks (Gal and Ghahramani, 2016).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROBABILITY_EPSILON = 1e-12

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


def mc_dropout_predictions(model: nn.Module, loader: DataLoader, n_samples: int, device: str) -> np.ndarray:
    """MC dropout probabilities of the bad class for every record of a loader.

    Args:
        model: Network returning logits.
        loader: Loader (not shuffled, so that rows follow the dataset order).
        n_samples: Stochastic passes.
        device: Device name.

    Returns:
        Array of shape (n_samples, records).
    """
    model.to(device)
    batches = [mc_dropout_probabilities(model, inputs.to(device), n_samples).cpu() for inputs, _ in loader]
    return torch.cat(batches, dim=1).numpy()


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    """Entropy (nats) of a Bernoulli variable.

    Args:
        probability: Probability of the positive outcome.

    Returns:
        Entropy, same shape as the input.
    """
    clipped = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))


def uncertainty_summary(samples: np.ndarray) -> pd.DataFrame:
    """Per-record summary of MC dropout samples.

    The predictive entropy (entropy of the mean probability) is split into the
    expected entropy of the single passes (aleatoric part) and their difference,
    the mutual information between prediction and weights (epistemic part).

    Args:
        samples: Probabilities of shape (n_samples, records).

    Returns:
        Table with ``probability_mean``, ``probability_std``, ``predictive_entropy``,
        ``aleatoric_entropy`` and ``mutual_information``.
    """
    mean = samples.mean(axis=0)
    predictive = binary_entropy(mean)
    aleatoric = binary_entropy(samples).mean(axis=0)
    return pd.DataFrame(
        {
            "probability_mean": mean,
            "probability_std": samples.std(axis=0),
            "predictive_entropy": predictive,
            "aleatoric_entropy": aleatoric,
            "mutual_information": predictive - aleatoric,
        }
    )
