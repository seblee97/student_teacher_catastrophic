from models.base_network import Model

import torch
import torch.distributions as tdist
import torch.nn as nn

from typing import Dict

class DriftingTeacher(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='teacher')
        self.drift_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([config.get(["task", "drift_size"])]))
        self.forward_count = 0

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is added to output
        """
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
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        return y
    
    def rotate_weights(self):
        for parameter_layer in self.state_dict():
            drift = self.drift_distribution.sample((len(self.state_dict()[parameter_layer]),))
            self.state_dict()[parameter_layer] = self.state_dict()[parameter_layer] + drift