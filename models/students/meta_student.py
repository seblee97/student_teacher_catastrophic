from .base_student import _BaseStudent

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

from typing import Dict

class MetaStudent(_BaseStudent):

    def __init__(self, config: Dict) -> None:

        _BaseStudent.__init__(self, config=config)

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
        pass

    def forward_all(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
        task_outputs = self._output_forward(x)
        return task_outputs

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        return y

    def _get_head_weights(self):
        import pdb; pdb.set_trace()
        raise NotImplementedError
