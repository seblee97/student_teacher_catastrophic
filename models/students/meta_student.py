from models.base_network import Model

import torch.nn as nn
import torch

from typing import Dict

class MetaStudent(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')

    def _construct_output_layers(self):

        self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        self._initialise_weights(self.output_layer)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def test_all_tasks(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / torch.sqrt(self.input_dimension))
        if self.task_setting == 'continual': 
            task_outputs = [self.heads[t](x) for t in range(self.num_teachers)]
        elif self.task_setting == 'meta':
            task_outputs = self.output_layer(x)
        return task_outputs

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        return y