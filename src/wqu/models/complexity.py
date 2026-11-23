"""Size and computational cost of a network: trainable parameters and floating-point operations.

The operations are counted on one forward pass with ``torch.utils.flop_counter.FlopCounterMode``,
which counts convolutions and matrix products (one multiply-add = 2 FLOPs) but not the fused
LSTM kernels of PyTorch. The LSTM layers are therefore counted separately, from their sizes and
from the length of the sequences they actually receive (forward hooks): for every time step,
direction and layer, the four gates cost ``4 * hidden * (input + hidden)`` multiply-adds. The
element-wise operations (activations, normalization, pooling, gate products) are not counted,
as is usual for FLOP counts of neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from wqu.models.diagnostics import count_parameters

FLOPS_PER_MULTIPLY_ADD = 2
LSTM_GATES = 4


@dataclass(frozen=True)
class ModelComplexity:
    """Size and cost of a network.

    Attributes:
        parameters: Trainable parameters.
        flops_per_record: Floating-point operations of one forward pass, per record.
        recurrent_flops_per_record: Part of ``flops_per_record`` due to the LSTM layers.
    """

    parameters: int
    flops_per_record: float
    recurrent_flops_per_record: float


def lstm_flops(module: nn.LSTM, batch_size: int, sequence_length: int) -> int:
    """Floating-point operations of the gate products of one LSTM call.

    Args:
        module: LSTM layer (stack of ``num_layers``, optionally bidirectional).
        batch_size: Sequences in the call.
        sequence_length: Time steps of each sequence.

    Returns:
        FLOPs of all layers, directions and time steps.

    Raises:
        ValueError: If the LSTM uses projections (not supported).
    """
    if module.proj_size:
        raise ValueError("LSTM layers with projections are not supported.")
    directions = 2 if module.bidirectional else 1
    hidden = module.hidden_size
    multiply_adds = 0
    for layer in range(module.num_layers):
        layer_input = module.input_size if layer == 0 else hidden * directions
        multiply_adds += LSTM_GATES * hidden * (layer_input + hidden) * directions
    return FLOPS_PER_MULTIPLY_ADD * multiply_adds * sequence_length * batch_size


def _lstm_call_shape(module: nn.LSTM, inputs: tuple) -> tuple[int, int]:
    """Batch size and sequence length of the input of an LSTM call.

    Raises:
        TypeError: If the input is not a tensor (e.g. a packed sequence).
    """
    sequence = inputs[0]
    if not isinstance(sequence, torch.Tensor):
        raise TypeError("Only LSTM calls on tensors are counted, not packed sequences.")
    if sequence.dim() == 2:
        return 1, int(sequence.shape[0])
    if module.batch_first:
        return int(sequence.shape[0]), int(sequence.shape[1])
    return int(sequence.shape[1]), int(sequence.shape[0])


def count_flops(model: nn.Module, inputs: torch.Tensor) -> tuple[int, int]:
    """Floating-point operations of one forward pass on a batch, in evaluation mode.

    The model keeps its device; the training mode is restored afterwards.

    Args:
        model: Network.
        inputs: Batch of inputs, on the device of the model.

    Returns:
        Tuple (total FLOPs of the batch, FLOPs of the LSTM layers in the batch).
    """
    recurrent = []

    def record_lstm(module: nn.LSTM, module_inputs: tuple, _output: object) -> None:
        recurrent.append(lstm_flops(module, *_lstm_call_shape(module, module_inputs)))

    handles = [module.register_forward_hook(record_lstm) for module in model.modules() if isinstance(module, nn.LSTM)]
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    recurrent_total = sum(recurrent)
    return counter.get_total_flops() + recurrent_total, recurrent_total


def model_complexity(model: nn.Module, inputs: torch.Tensor) -> ModelComplexity:
    """Trainable parameters and FLOPs per record of a network.

    Args:
        model: Network.
        inputs: Batch of inputs, on the device of the model.

    Returns:
        ModelComplexity.
    """
    total, recurrent = count_flops(model, inputs)
    batch_size = int(inputs.shape[0])
    return ModelComplexity(count_parameters(model), total / batch_size, recurrent / batch_size)
