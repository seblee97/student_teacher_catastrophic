"""Adapted from implementation by moskomule

https://github.com/moskomule/ewc.pytorch
"""
import copy

import constants
import torch
from regularisers import base_regulariser
from torch import nn
from torch.nn import functional as F


class NodeConsolidation(base_regulariser.BaseRegulariser):
    def compute_first_task_importance(
        self,
        student: nn.Module,
        previous_teacher_index: int,
        previous_teacher: nn.Module,
        loss_function,
        data_module,
    ):
        self._student = student
        self._previous_teacher = previous_teacher
        self._previous_teacher_index = previous_teacher_index
        self._new_teacher_index = self._student.current_teacher

        self._loss_function = loss_function
        self._dataset = data_module.get_test_data()[constants.Constants.X].to(
            self._device
        )

        self._params = {
            n: p for n, p in self._student.named_parameters() if "heads" not in n
        }
        self._store_previous_task_parameters()

        self._precision_matrices = self._diag_fisher()

    @property
    def precision_matrices(self):
        return self._precision_matrices

    def _diag_fisher(self):
        # to compute Fischer on previous task, switch heads
        self._student.signal_task_boundary(new_task=self._previous_teacher_index)

        # limit to two layer networks
        first_layer_param_copy = list(copy.deepcopy(self._params).values())[0]
        first_layer_name = list(self._params.keys())[0]

        node_fischer = {first_layer_name: torch.zeros(first_layer_param_copy.shape[0])}

        self._student.eval()
        for data in self._dataset:
            self._student.zero_grad()

            # get post activations
            pre_activation = self._student.layers[0](data)
            post_activation = self._student.nonlinear_function(pre_activation)
            output = (
                self._student._forward_scaling
                * self._student._get_output_from_head(post_activation)
            )
            label = self._previous_teacher(data)
            loss = self._loss_function(output, label)

            # get derivative of loss wrt post-activation
            derivative = torch.autograd.grad(loss, post_activation)[0]

            for node_index, node_derivative in enumerate(derivative):
                node_fischer[first_layer_name][
                    node_index
                ] += node_derivative ** 2 / len(self._dataset)

        # return back head
        self._student.signal_task_boundary(new_task=self._new_teacher_index)

        return node_fischer

    def penalty(self, student: nn.Module):
        loss = 0
        for n, param in student.named_parameters():
            if "heads" not in n:

                squared_parameter_difference = (
                    param - self._previous_task_parameters[0][n]
                ) ** 2
                
                for m in range(len(self._precision_matrices[n])):
                    loss += (
                        self._precision_matrices[n][m] * squared_parameter_difference[m]
                    ).sum()

        return self._importance * loss
