import abc
import copy
import math
import random
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import constants
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
from utils import base_network
from utils import custom_activations


class BaseTeacher(base_network.BaseNetwork, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        forward_hidden_scaling: float,
        forward_scaling: float,
        nonlinearity: str,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_std: Union[float, int],
        initialisation_std: Optional[float] = None,
        zero_head: bool = False,
    ) -> None:
        self._unit_norm_teacher_head = unit_norm_teacher_head
        self._noise_std = noise_std
        self._zero_head = zero_head

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            nonlinearity=nonlinearity,
            weight_normalisation=weight_normalisation,
            initialisation_std=initialisation_std,
        )

        self._freeze()

    @property
    def head(self) -> nn.Linear:
        return self._head

    def _freeze(self) -> None:
        for layer in self._layers:
            for param in layer.parameters():
                param.requires_grad = False

    def _construct_output_layers(self):
        self._head = nn.Linear(
            self._hidden_dimensions[-1], self._output_dimension, bias=self._bias
        )
        if self._unit_norm_teacher_head:
            head_norm = torch.norm(self._head.weight)

            normalised_head = self._head.weight / head_norm

            self._head.weight.data = normalised_head
        else:
            self._initialise_weights(self._head)

        if self._zero_head:
            self._head.weight.data = torch.zeros_like(self._head.weight.data)

        for param in self._head.parameters():
            param.requires_grad = False

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant teacher head."""
        pass
