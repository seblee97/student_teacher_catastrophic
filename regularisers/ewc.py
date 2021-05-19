"""Adapted from implementation by moskomule

https://github.com/moskomule/ewc.pytorch
"""
import copy

import constants

import torch
from torch import nn
from torch.nn import functional as F
# from torch.autograd import Variable


class EWC:
    def __init__(self, student: nn.Module, previous_teacher_index: int, previous_teacher: nn.Module, loss_function, data_module, device: str):

        self._student = student
        self._previous_teacher = previous_teacher

        # to compute Fischer on previous task, switch heads
        self._student.signal_task_boundary(new_task=previous_teacher_index)

        self._device = device
        self._loss_function = loss_function
        self._dataset = data_module.get_test_data()[constants.Constants.X].to(self._device)

        self._params = {n: p for n, p in self._student.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, param in copy.deepcopy(self._params).items():
            self._means[n] = param.data.to(self._device)

    def _diag_fisher(self):
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
                if param.requires_grad:
                    precision_matrices[n].data += param.grad.data ** 2 / len(self._dataset)

        precision_matrices = {n: param for n, param in precision_matrices.items()}

        return precision_matrices

    def penalty(self, student: nn.Module):
        loss = 0
        for n, param in student.named_parameters():
            if param.requires_grad:
                _loss = self._precision_matrices[n] * (param - self._means[n]) ** 2
                loss += _loss.sum()
        return loss