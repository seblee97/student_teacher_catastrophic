import copy
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from utils import custom_activations


def get_nonlinear_function(nonlinearity_name: str) -> Callable:
    """Makes the nonlinearity function specified by the config.

    Args:
        nonlinearity_name: name of activation function or non-linearity.

    Returns:
        nonlinear_function: Callable object, the nonlinearity function

    Raises:
        ValueError: If nonlinearity name provided is not recognised
    """
    if nonlinearity_name == constants.Constants.RELU:
        nonlinear_function = F.relu
    elif nonlinearity_name == constants.Constants.SIGMOID:
        nonlinear_function = torch.sigmoid
    elif nonlinearity_name == constants.Constants.SCALED_ERF:
        nonlinear_function = custom_activations.scaled_erf_activation
    elif nonlinearity_name == constants.Constants.LINEAR:
        nonlinear_function = custom_activations.linear_activation
    else:
        raise ValueError(f"Unknown non-linearity: {nonlinearity_name}")
    return nonlinear_function


def construct_layers(
    input_dimension: int, hidden_dimensions: List[int], bias: bool
) -> nn.ModuleList:
    """Instantiate layers (input, hidden).

    Args:
        input_dimension: size of input.
        hidden_dimension: list of integer dimensions for hidden layers.
        bias: whether or not to include bias parameters.

    Returns:
        layers: nn module list of weights.
    """
    layers = nn.ModuleList([])

    layer_dimensions = [input_dimension] + hidden_dimensions

    for layer_size, next_layer_size in zip(layer_dimensions, layer_dimensions[1:]):
        layer = nn.Linear(layer_size, next_layer_size, bias=bias)
        initialise_weights(layer, initialisation_std, bias, symmetric_initialisation)
        layers.append(layer)

    return layers


def initialise_weights(
    layer: nn.Module,
    bias: bool,
    initialisation_std: Optional[float] = None,
    symmetric_initialisation: bool = None,
) -> None:
    """In-place weight initialisation for a given layer in accordance with configuration.

    Args:
        layer: the layer to be initialised.
    """
    if initialisation_std is not None:
        nn.init.normal_(layer.weight, std=initialisation_std)
        if bias:
            nn.init.normal_(layer.bias, std=initialisation_std)
    if symmetric_initialisation:
        # copy first component of weight matrix to others to ensure zero overlap
        base_parameters = copy.deepcopy(
            layer.state_dict()[constants.Constants.WEIGHT][0]
        )
        for dim in range(1, len(layer.state_dict()[constants.Constants.WEIGHT])):
            layer.state_dict()[constants.Constants.WEIGHT][dim] = base_parameters
