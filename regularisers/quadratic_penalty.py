import copy

import constants

import torch
from torch import nn
from torch.nn import functional as F

from regularisers import base_regulariser


class QuadraticPenalty(base_regulariser.BaseRegulariser):

    def compute_first_task_importance(
        self, 
        student: nn.Module, 
        previous_teacher_index: int, 
        previous_teacher: nn.Module, 
        loss_function,
        data_module,
    ):
        self._params = {n: p for n, p in student.named_parameters() if "heads" not in n}
        self._store_previous_task_parameters()

    def penalty(self, student: nn.Module):
        loss = 0
        for n, param in student.named_parameters():
            if "heads" not in n:
                _loss = (param - self._previous_task_parameters[0][n]) ** 2
                loss += _loss.sum()
        return self._importance * loss