"""Class-weighted binary cross-entropy and correction of the resulting probabilities.

Weighting the bad class by ``w = n_good / n_bad`` trains the network as if the
classes were balanced, so its probabilities refer to a balanced prior. They are
mapped back to the class frequencies of the data with ``prior_corrected_probability``.
Without weighting (``training.class_weighting: none``) the weight is 1 and the correction
leaves the probabilities unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from wqu.config import BALANCED_CLASS_WEIGHTING, NO_CLASS_WEIGHTING

UNWEIGHTED = 1.0


def bad_class_weight(labels: np.ndarray) -> float:
    """Weight of the bad class (label 1) that balances the two classes.

    Args:
        labels: Training labels (0 = good, 1 = bad).

    Returns:
        Number of good records divided by number of bad records.

    Raises:
        ValueError: If one of the classes is absent.
    """
    n_bad = int(np.sum(labels == 1))
    n_good = int(np.sum(labels == 0))
    if n_bad == 0 or n_good == 0:
        raise ValueError("Both classes must be present to compute the class weight.")
    return n_good / n_bad


def class_weight(labels: np.ndarray, weighting: str) -> float:
    """Weight of the bad class for a weighting scheme.

    Args:
        labels: Training labels (0 = good, 1 = bad).
        weighting: ``balanced`` (``bad_class_weight``) or ``none`` (weight 1).

    Returns:
        Weight of the bad class.

    Raises:
        ValueError: If the scheme is unknown.
    """
    if weighting == BALANCED_CLASS_WEIGHTING:
        return bad_class_weight(labels)
    if weighting == NO_CLASS_WEIGHTING:
        return UNWEIGHTED
    raise ValueError(f"Unknown class weighting '{weighting}'.")


def weighted_bce_loss(bad_weight: float) -> nn.BCEWithLogitsLoss:
    """Binary cross-entropy on logits with the bad class weighted.

    Args:
        bad_weight: Weight of the bad class (see ``bad_class_weight``).

    Returns:
        Loss module.
    """
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bad_weight))


def prior_corrected_probability(probability: np.ndarray, bad_weight: float) -> np.ndarray:
    """Map probabilities of a network trained with class weight ``w`` back to the data prior.

    With the bad class weighted by ``w``, the trained network estimates
    ``q = w p / (w p + 1 - p)``, where ``p`` is the probability under the data
    prior; inverting gives ``p = q / (q + w (1 - q))``.

    Args:
        probability: Probabilities of the bad class from the weighted network.
        bad_weight: Weight used in training.

    Returns:
        Probabilities of the bad class under the class frequencies of the training data.
    """
    return probability / (probability + bad_weight * (1.0 - probability))
