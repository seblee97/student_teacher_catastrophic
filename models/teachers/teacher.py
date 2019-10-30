from models.base_network import Model

import torch
import torch.distributions as tdist
import torch.nn as nn

from typing import Dict

class Teacher(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='teacher')

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is added to output
        """
        if mean is None:
            self.noise_distribution = None
        else:
            self.noise_distribution = tdist.Normal(torch.Tensor([mean]), torch.Tensor([std]))

    def _construct_output_layers(self):
        self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        self._initialise_weights(self.output_layer)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        if self.noise_distribution:
            noise = self.noise_distribution.sample((self.batch_dimension,))
            y = y + noise
        return y