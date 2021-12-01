import abc
import copy

from torch import nn


class BaseRegulariser(abc.ABC):
    def __init__(self, importance: float, device: str):
        self._importance = importance
        self._device = device

        self._previous_task_parameters = []

    def _store_previous_task_parameters(self):
        """Store the parameters from before task switches."""
        previous_task_paramters = {}
        for n, param in copy.deepcopy(self._params).items():
            previous_task_paramters[n] = param.data.to(self._device)

        self._previous_task_parameters.append(previous_task_paramters)

    @abc.abstractmethod
    def compute_first_task_importance(
        self,
        student: nn.Module,
        previous_teacher_index: int,
        previous_teacher: nn.Module,
        loss_function,
        data_module,
        device: str,
    ):
        pass

    @abc.abstractmethod
    def penalty(self, student: nn.Module):
        pass

    @property
    def previous_task_parameters(self):
        return self._previous_task_parameters[-1]
