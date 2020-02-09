from models.base_network import Model

import torch.nn as nn
import torch

import numpy as np

from typing import Dict

class MetaStudent(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')

    def _construct_output_layers(self):

        self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        if self.initialise_student_outputs:
            self.output_layer = self._initialise_weights(self.output_layer)
        else:
            torch.nn.init.zeros_(self.output_layer.weight)
            if self.bias:
                torch.nn.init.zeros_(self.output_layer.bias)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def set_task(self, task_index: int):
        self._current_teacher = task_index

    def test_all_tasks(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
        task_outputs = self.output_layer(x)
        return task_outputs

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        return y

    def _get_head_weights(self):
        import pdb; pdb.set_trace()
        raise NotImplementedError

class MNISTMetaStudent(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')

    def _construct_output_layers(self):
        raise NotImplementedError

    def set_task(self, task_index: int):
        raise NotImplementedError

    def test_all_tasks(self, x: torch.Tensor):
        raise NotImplementedError

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_head_weights(self):
        raise NotImplementedError
