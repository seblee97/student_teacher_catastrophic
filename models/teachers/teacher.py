from models.base_network import Model

import torch
import torch.distributions as tdist
import torch.nn as nn

from typing import Dict

class Teacher(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='teacher')
        self.noisy = False

    def get_output_statistics(self, repeats=1000):
        with torch.no_grad():
            random_input = torch.randn(repeats, self.input_dimension)
            output = self.forward(random_input)
            output_mean = float(torch.mean(output))
            output_std = float(torch.std(output))
            output_max = float(max(output))
            output_min = float(min(output))
            output_range = output_max - output_min

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is added to output
        """
        self.noise_distribution = tdist.Normal(torch.Tensor([mean]), torch.Tensor([std]))
        self.noisy = True

    def _construct_output_layers(self):
        self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        self._initialise_weights(self.output_layer)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        if self.noisy:
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        return y