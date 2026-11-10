"""Grad-CAM maps of the classifier, with MC dropout.

Grad-CAM (Selvaraju et al., 2017) weights the activation maps of a convolutional
layer by the spatial mean of the gradient of the class score and keeps the positive
part of their sum. It is computed with Captum (``captum.attr.LayerGradCam``) on the
last convolutional layer of the network, before the recurrent layers.

The network has one output (the logit of the bad class): the map for ``bad`` uses
the logit, the map for ``good`` uses its opposite. Each map is computed on
``n_samples`` stochastic passes with dropout active (MC dropout): their mean is the
explanation, their standard deviation shows where the explanation itself is uncertain.
The maps are interpolated to the frequency-time grid of the input; they are common
to the three components, because Grad-CAM sums over the channels of the layer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from captum.attr import LayerAttribution, LayerGradCam
from torch import nn

from wqu.models.uncertainty import enable_mc_dropout

CLASS_SIGNS: dict[str, float] = {"bad": 1.0, "good": -1.0}
INTERPOLATION_MODE = "bilinear"


@dataclass(frozen=True)
class GradCamResult:
    """Grad-CAM maps of one class over MC dropout passes.

    Attributes:
        mean: Mean map, shape (batch, frequencies, times).
        std: Standard deviation across passes, shape (batch, frequencies, times).
        samples: Map of every pass, shape (n_samples, batch, frequencies, times).
    """

    mean: torch.Tensor
    std: torch.Tensor
    samples: torch.Tensor


def class_score(model: nn.Module, class_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Function returning the score of a class, to be explained by Grad-CAM.

    Args:
        model: Network returning the logit of the bad class, shape (batch,).
        class_name: ``bad`` or ``good``.

    Returns:
        Function mapping an input batch to scores of shape (batch,).

    Raises:
        ValueError: If the class name is unknown.
    """
    if class_name not in CLASS_SIGNS:
        raise ValueError(f"Unknown class '{class_name}'; expected one of {tuple(CLASS_SIGNS)}.")
    sign = CLASS_SIGNS[class_name]
    return lambda inputs: sign * model(inputs)


def grad_cam(
    model: nn.Module, layer: nn.Module, inputs: torch.Tensor, class_name: str, n_samples: int
) -> GradCamResult:
    """Grad-CAM maps of one class over MC dropout passes, interpolated to the input grid.

    The model is left in MC dropout mode (dropout active, batch normalization frozen).
    cuDNN is disabled during the computation, because its recurrent layers cannot be
    differentiated outside training mode on CUDA devices.

    Args:
        model: Network returning the logit of the bad class.
        layer: Convolutional layer whose activations are weighted.
        inputs: Batch of shape (batch, components, frequencies, times).
        class_name: ``bad`` or ``good``.
        n_samples: Stochastic passes.

    Returns:
        GradCamResult.
    """
    enable_mc_dropout(model)
    explainer = LayerGradCam(class_score(model, class_name), layer)
    size = tuple(inputs.shape[-2:])
    maps = []
    with torch.backends.cudnn.flags(enabled=False):
        for _ in range(n_samples):
            attribution = explainer.attribute(inputs, relu_attributions=True)
            maps.append(LayerAttribution.interpolate(attribution, size, interpolate_mode=INTERPOLATION_MODE)[:, 0])
    samples = torch.stack(maps).detach()
    return GradCamResult(mean=samples.mean(dim=0), std=samples.std(dim=0), samples=samples)


def grad_cam_by_class(
    model: nn.Module, layer: nn.Module, inputs: torch.Tensor, n_samples: int
) -> dict[str, GradCamResult]:
    """Grad-CAM maps of both classes.

    Args:
        model: Network returning the logit of the bad class.
        layer: Convolutional layer whose activations are weighted.
        inputs: Batch of shape (batch, components, frequencies, times).
        n_samples: Stochastic passes per class.

    Returns:
        Mapping from class name to GradCamResult.
    """
    return {class_name: grad_cam(model, layer, inputs, class_name, n_samples) for class_name in CLASS_SIGNS}


def time_profile(grad_cam_map: torch.Tensor) -> torch.Tensor:
    """Mean of a map over frequencies, to compare it with the waveform.

    Args:
        grad_cam_map: Map of shape (..., frequencies, times).

    Returns:
        Tensor of shape (..., times).
    """
    return grad_cam_map.mean(dim=-2)
