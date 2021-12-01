"""Adapted from implementation by moskomule

https://github.com/moskomule/ewc.pytorch
"""
import copy

from cata import constants
from cata.regularisers import base_regulariser
from torch import nn


class EWC(base_regulariser.BaseRegulariser):
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
        self._dataset = data_module.get_test_data()[constants.X].to(self._device)

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

        precision_matrices = {}

        for n, param in copy.deepcopy(self._params).items():
            param.data.zero_()
            precision_matrices[n] = param.data.to(self._device)

        self._student.eval()
        for data in self._dataset:
            self._student.zero_grad()
            output = self._student(data)
            label = self._previous_teacher(data)
            loss = self._loss_function(output, label)
            loss.backward()

            for n, param in self._student.named_parameters():
                if "heads" not in n:
                    precision_matrices[n].data += param.grad.data ** 2 / len(
                        self._dataset
                    )

        precision_matrices = {n: param for n, param in precision_matrices.items()}

        # return back head
        self._student.signal_task_boundary(new_task=self._new_teacher_index)

        return precision_matrices

    def penalty(self, student: nn.Module):
        loss = 0
        for n, param in student.named_parameters():
            if "heads" not in n:
                _loss = (
                    self._precision_matrices[n]
                    * (param - self._previous_task_parameters[0][n]) ** 2
                )
                loss += _loss.sum()
        return self._importance * loss
