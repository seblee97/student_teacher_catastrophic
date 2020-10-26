import abc
import copy
import math
import random
from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

import constants
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
        nonlinearity: str,
        unit_norm_teacher_head: bool,
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
            initialisation_std=initialisation_std,
        )

    # def get_output_statistics(self, repeats=5000):
    #     with torch.no_grad():
    #         random_input = torch.randn(repeats, self.input_dimension)
    #         output = self.forward(random_input)
    #         output_std = float(torch.std(output))
    #     return output_std

    # def add_output_noise(self, clean_tensor: torch.Tensor) -> torch.Tensor:
    #     noise_sample = self.noise_distribution.sample(clean_tensor.shape)
    #     return clean_tensor + noise_sample

    # def set_noise_distribution(self, mean: float, std: float):
    #     """
    #     Sets a normal distribution from which to sample noise that is
    #     added to output
    #     """
    #     if std == 0:
    #         raise ValueError("Standard Deviation of Normal cannot be 0.")
    #     self.noise_distribution = tdist.Normal(mean, std)
    #     self.noisy = True

    # def load_weights(self, weights_path: str) -> None:
    #     """Load saved weights into model"""
    #     torch.load(weights_path)
    #     self.load_state_dict(torch.load(weights_path))

    def _construct_output_layers(self):
        self._head = nn.Linear(
            self._hidden_dimensions[-1], self._output_dimension, bias=self._bias
        )
        if self._unit_norm_teacher_head:
            head_weight = random.choice([1, -1])
            self._head.weight.data.fill_(head_weight)
        else:
            self._initialise_weights(self._head)
        for param in self._head.parameters():
            param.requires_grad = False

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant teacher head."""
        pass
