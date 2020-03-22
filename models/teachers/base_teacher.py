from models.base_network import Model

import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Dict

from abc import ABC, abstractmethod

class _Teacher(Model, ABC):

    def __init__(self, config: Dict, index: int) -> None:

        Model.__init__(self, config=config, model_type='teacher_{}'.format(str(index)))
        self.noisy = False

    def get_output_statistics(self, repeats=5000):
        with torch.no_grad():
            random_input = torch.randn(repeats, self.input_dimension)
            output = self.forward(random_input)
            output_mean = float(torch.mean(output))
            output_std = float(torch.std(output))
            output_max = float(max(output))
            output_min = float(min(output))
            output_range = output_max - output_min
        return output_std

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is added to output
        """
        if std == 0:
            raise ValueError("Standard Deviation of Normal cannot be 0.")
        self.noise_distribution = tdist.Normal(torch.Tensor([mean]), torch.Tensor([std]))
        self.noisy = True
    
    def load_weights(self, weights_path: str) -> None:
        """Load saved weights into model"""
        sdtet = torch.load(weights_path)
        self.load_state_dict(torch.load(weights_path))

    def _construct_output_layers(self):
        self.output_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias))
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    @abstractmethod
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")