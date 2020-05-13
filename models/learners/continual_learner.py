import torch.nn as nn
import torch
import torch.nn.functional as F

import numpy as np

from .base_learner import _BaseLearner

from typing import Dict, List

class ContinualLearner(_BaseLearner):

    """Orchestrates students in continual learning setting (one head per task)"""

    def __init__(self, config):
        _BaseLearner.__init__(self, config)

        self._current_teacher = None

    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        self.set_task(new_task)

    def signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

    def _construct_output_layers(self):
        # create one head per teacher
        self.heads = nn.ModuleList([])
        for _ in range(self.num_teachers):
            output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
            if self.initialise_student_outputs:
                output_layer = self._initialise_weights(output_layer)
            else:
                torch.nn.init.zeros_(output_layer.weight)
                if self.bias:
                    torch.nn.init.zeros_(output_layer.bias)
            # freeze heads (unfrozen when current task)
            for param in output_layer.parameters():
                param.requires_grad = False
            self.heads.append(output_layer)

    def set_task(self, task_index: int):
        # freeze weights of head for previous task
        if self._current_teacher:
            for param in self.heads[self._current_teacher].parameters():
                param.requires_grad = False
        # unfreeze weights of head for new task
        for param in self.heads[task_index].parameters():
            param.requires_grad = True
        self._current_teacher = task_index

    # def forward_with_head(self, x: torch.Tensor, task_index: int):
    #     for layer in self.layers:
    #         x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
    #     task_output = self.heads[task_index](x)
    #     if self.classification_output:
    #         task_output = self._threshold(task_output)
    #     return task_output

    def forward_batch_per_task(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        assert len(batch_list) == self.num_teachers, \
            "Forward of one batch per head requires equal number of batches to heads"
        
        outputs = []

        for task_index, batch in enumerate(batch_list):
            x = batch['x']
            for layer in self.layers:
                x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
            task_output = self.heads[task_index](x)
            if self.classification_output:
                task_output = self._threshold(task_output)
            outputs.append(task_output)
        
        return outputs
 
    def forward_all(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
        task_outputs = [self.heads[t](x) for t in range(self.num_teachers)]
        if self.classification_output:
            return [self._threshold(t) for t in task_outputs]
        return task_outputs

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        y = self.heads[self._current_teacher](x)
        return y

    def _get_head_weights(self) -> List[torch.Tensor]:
        # extract weights of output layer. Multi-headed for continual setting => one set per teacher
        head_weights = [self.state_dict()['heads.{}.weight'.format(h)] for h in range(self.num_teachers)] # could use self.heads directly
        return head_weights
