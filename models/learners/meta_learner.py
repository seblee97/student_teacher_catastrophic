from .base_learner import _BaseLearner

from typing import Dict, List, Iterator

import torch.nn as nn
import torch


class MetaLearner(_BaseLearner):

    """Orchestrates students in meta learning setting (one head only)"""

    def __init__(self, config):
        _BaseLearner.__init__(self, config)

    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        self.set_task(new_task)

    def signal_step_boundary_to_learner(
        self,
        step: int,
        current_task: int
    ) -> None:
        pass

    def _construct_output_layers(self):

        self.output_layer = nn.Linear(
            self.hidden_dimensions[-1], self.output_dimension, bias=self.bias
            )
        if self.initialise_student_outputs:
            self.output_layer = self._initialise_weights(self.output_layer)
        else:
            torch.nn.init.zeros_(self.output_layer.weight)
            if self.bias:
                torch.nn.init.zeros_(self.output_layer.bias)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def _get_trainable_head_parameters(
        self
    ) -> List[Dict[str, Iterator[torch.nn.Parameter]]]:
        trainable_head_parameters = [{
                'params': self.output_layer.parameters(),
                'lr': self._learning_rate / self._input_dimension
            }]
        return trainable_head_parameters

    def set_task(self, task_index: int):
        pass

    def forward_batch_per_task(
        self,
        batch_list: List[Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward_all(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(
                self.forward_scaling * layer(x)
                )
        task_outputs = self._output_forward(x)
        all_task_outputs = [task_outputs for _ in range(self._num_teachers)]
        return all_task_outputs

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        return y

    def _get_head_weights(self) -> List[torch.Tensor]:
        """
        This method extracts the weights of the outputs layer.
        For the meta learning setting there is only one head regardless
        of the number of teachers.

        Returns:
            head_weights: list of torch tensors that are the weights
            of the heads.
        """
        head_weights = [self.output_layer.state_dict()['weight']]
        return head_weights
