import abc
import math
import copy
from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from utils import custom_activations


class BaseNetwork(nn.Module, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        forward_scaling: float,
        symmetric_initialisation: Optional[bool] = False,
        initialisation_std: Optional[float] = None,
    ) -> None:
        super().__init__()

        self._input_dimension = input_dimension
        self._hidden_dimensions = hidden_dimensions
        self._output_dimension = output_dimension
        self._bias = bias
        self._classification_output = loss_type == constants.Constants.CLASSIFICATION
        self._nonlinearity = nonlinearity
        self._initialisation_std = initialisation_std
        self._symmetric_initialisation = symmetric_initialisation
        self._forward_scaling = forward_scaling

        self._nonlinear_function = self._get_nonlinear_function()
        self._construct_layers()

    @property
    def layers(self) -> nn.ModuleList:
        return self._layers

    @property
    def self_overlap(self):
        with torch.no_grad():
            layer = self._layers[0].weight.data
            overlap = layer.mm(layer.t()) / self._input_dimension
        return overlap

    def _get_nonlinear_function(self) -> Callable:
        """Makes the nonlinearity function specified by the config.

        Returns:
            nonlinear_function: Callable object, the nonlinearity function

        Raises:
            ValueError: If nonlinearity provided in config is not recognised
        """
        if self._nonlinearity == constants.Constants.RELU:
            nonlinear_function = F.relu
        elif self._nonlinearity == constants.Constants.SIGMOID:
            nonlinear_function = torch.sigmoid
        elif self._nonlinearity == constants.Constants.SCALED_ERF:
            nonlinear_function = custom_activations.scaled_erf_activation
        elif self._nonlinearity == constants.Constants.LINEAR:
            nonlinear_function = custom_activations.linear_activation
        else:
            raise ValueError(f"Unknown non-linearity: {self._nonlinearity}")
        return nonlinear_function

    def _construct_layers(self) -> None:
        """Instantiate layers (input, hidden and output) according to
        dimensions specified in configuration. Note this method makes a call to
        the abstract _construct_output_layers, which is implemented by the
        child.
        """
        self._layers = nn.ModuleList([])

        layer_dimensions = [self._input_dimension] + self._hidden_dimensions

        for layer_size, next_layer_size in zip(layer_dimensions, layer_dimensions[1:]):
            layer = nn.Linear(layer_size, next_layer_size, bias=self._bias)
            self._initialise_weights(layer)
            self._layers.append(layer)

        self._construct_output_layers()

    def _initialise_weights(self, layer: nn.Module) -> None:
        """In-place weight initialisation for a given layer in accordance with configuration.

        Args:
            layer: the layer to be initialised.
        """
        if self._initialisation_std is not None:
            nn.init.normal_(layer.weight, std=self._initialisation_std)
            if self._bias:
                nn.init.normal_(layer.bias, std=self._initialisation_std)
        if self._symmetric_initialisation:
            # copy first component of weight matrix to others to ensure zero overlap
            base_parameters = copy.deepcopy(
                layer.state_dict()[constants.Constants.WEIGHT][0]
            )
            for dim in range(1, len(layer.state_dict()[constants.Constants.WEIGHT])):
                layer.state_dict()[constants.Constants.WEIGHT][dim] = base_parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method performs the forward pass. This implements the
        abstract method from the nn.Module base class.

        Args:
            x: input tensor

        Returns:
            y: output of network
        """
        for layer in self._layers:
            x = self._nonlinear_function(self._forward_scaling * layer(x))

        y = self._get_output_from_head(x)

        return y

    @abc.abstractmethod
    def _construct_output_layers(self):
        """Instantiate the output layer."""
        pass

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head."""
        pass
