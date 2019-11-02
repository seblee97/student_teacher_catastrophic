from models.base_network import Model

import torch.nn as nn
import torch

import numpy as np

from typing import Dict

class ContinualStudent(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')

    def _construct_output_layers(self):

        # create one head per teacher
        self.heads = nn.ModuleList([])
        for _ in range(self.num_teachers):
            output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
            self._initialise_weights(output_layer)
            if self.soft_committee:
                for param in output_layer.parameters():
                    param.requires_grad = False
            self.heads.append(output_layer)

    def set_task(self, task_index: int):
        self._current_teacher = task_index

    def test_all_tasks(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
        task_outputs = [self.heads[t](x) for t in range(self.num_teachers)]
        return task_outputs

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.heads[self._current_teacher](x)
        return y