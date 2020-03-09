from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from typing import Dict, List

class _BaseTeacher(ABC):

    def __init__(self, config: Dict):

        self._num_teachers = config.get(["task", "num_teachers"])
        self._device = config.get("device")

        self._setup_teachers(config=config)

    @abstractmethod
    def _setup_teachers(self, config: Dict) -> None:
        """instantiate teacher network(s)"""
        raise NotImplementedError("Base class method")

    def forward(self, teacher_index: int, x: torch.Tensor) -> torch.Tensor:
        """call to current teacher forward"""
        output = self._teachers[teacher_index](x)
        return output

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """call to forward of all teachers (used primarily for evaluation)"""
        outputs = [teacher(x) for teacher in self._teachers]
        return outputs

    def get_teacher_networks(self) -> List[nn.Module]:
        """getter method for teacher networks"""
        networks = self._teachers 
        return networks

    @abstractmethod
    def signal_task_boundary_to_teacher(self, new_task: int) -> None:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def signal_step_boundary_to_teacher(self, step: int, current_task: int) -> None:
        raise NotImplementedError("Base class method")


