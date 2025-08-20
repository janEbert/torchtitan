import torch
from typing import Callable


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) * x


ACTIVATION_FUNCTIONS = {
    "silu": torch.nn.functional.silu,
    "relu_squared": relu_squared,
    "relu_square": relu_squared,
    "elu": torch.nn.functional.elu,
    "relu": torch.nn.functional.relu,
    "selu": torch.nn.functional.selu,
    "gelu": lambda x: x * torch.nn.functional.sigmoid(1.702 * x),
    "sigmoid": torch.nn.functional.sigmoid,
}


def build_activation(activation_type: str) -> Callable:
    """
    Builds the specified activation layer based on the activation_type.

    Args:
        activation_type (str): The type of activation layer to build.

    Returns:
        The built activation layer.

    Raises:
        NotImplementedError: If an unknown activation_type is provided.
    """
    activation_type = activation_type.lower()  # Normalize to lowercase

    activation_layer_fn = ACTIVATION_FUNCTIONS.get(activation_type)
    if activation_layer_fn is not None:
        return activation_layer_fn
    else:
        raise NotImplementedError(f"Unknown activation_type: '{activation_type}'")
