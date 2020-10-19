import abc
import copy
import math
from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist

import constants
from utils import custom_activations
from students import base_network


class BaseTeacher(base_network.BaseNetwork, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        soft_committee: bool,
        loss_type: str,
        nonlinearity: str,
        initialise_outputs: bool,
        unit_norm_teacher_head: bool,
        symmetric_initialisation: bool = False,
        initialisation_std: Optional[float] = None,
    ) -> None:
        self._unit_norm_teacher_head = unit_norm_teacher_head

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            initialise_outputs=initialise_outputs,
            symmetric_initialisation=symmetric_initialisation,
            initialisation_std=initialisation_std,
        )

    def get_output_statistics(self, repeats=5000):
        with torch.no_grad():
            random_input = torch.randn(repeats, self.input_dimension)
            output = self.forward(random_input)
            output_std = float(torch.std(output))
        return output_std

    def add_output_noise(self, clean_tensor: torch.Tensor) -> torch.Tensor:
        noise_sample = self.noise_distribution.sample(clean_tensor.shape)
        return clean_tensor + noise_sample

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is
        added to output
        """
        if std == 0:
            raise ValueError("Standard Deviation of Normal cannot be 0.")
        self.noise_distribution = tdist.Normal(mean, std)
        self.noisy = True

    def load_weights(self, weights_path: str) -> None:
        """Load saved weights into model"""
        torch.load(weights_path)
        self.load_state_dict(torch.load(weights_path))

    def _construct_output_layers(self):
        self.output_layer = nn.Linear(
            self.hidden_dimensions[-1], self.output_dimension, bias=self.bias
        )
        if self.unit_norm_teacher_head:
            head_weight = random.choice([1, -1])
            self.output_layer.weight.data.fill_(head_weight)
        else:
            self._initialise_weights(self.output_layer)
        for param in self.output_layer.parameters():
            param.requires_grad = False

    @abc.abstractmethod
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")
