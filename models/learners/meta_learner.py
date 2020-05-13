from models.networks.students import MetaStudent

from .base_learner import _BaseLearner

from typing import Dict, List

import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

class MetaLearner(_BaseLearner):

    """Orchestrates students in meta learning setting (one head only)"""

    def __init__(self, config):
        _BaseLearner.__init__(self, config)

    def _setup_student(self, config: Dict) -> None:
        """Instantiate student"""
        self._student_network = MetaStudent(config=config).to(self._device)

    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        self._student_network.set_task(new_task)

    def signal_step_boundary_to_learner(self, step: int, current_task: int) -> None:
        pass

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

    def forward_batch_per_task(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        raise NotImplementedError

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
