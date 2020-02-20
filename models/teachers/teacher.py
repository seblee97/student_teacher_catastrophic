from models.base_network import Model

import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Dict

class Teacher(Model):

    """Regression"""

    def __init__(self, config: Dict, index: int) -> None:

        Model.__init__(self, config=config, model_type='teacher_{}'.format(str(index)))
        self.noisy = False

        if config.get(["task", "loss_type"]) == "classification":
            self.classification_output = True
        elif config.get(["task", "loss_type"]) == 'regression':
            self.classification_output = False
        else:
            raise ValueError("Unknown loss type given in base config")

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

    def _construct_output_layers(self):
        self.output_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias))
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        if self.noisy:
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        if self.classification_output: # think about taking this into separate teacher class as below
            sigmoid_y = F.sigmoid(y)
            return (sigmoid_y > 0.5).type(torch.LongTensor).reshape(len(sigmoid_y),)
        return y


class ClassificationTeacher(Teacher):

    """Classification - sign output"""

    def __init__(self, config: Dict, index: int) -> None:

        Teacher.__init__(self, config=config, model_type='teacher_{}'.format(str(index)))

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override teacher output forward, add sign output
        """
        y = self.output_layer(x)
        if self.noisy:
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        # y = np.sign(y)
        if y > 0:
            return 1
        else:
            return 0