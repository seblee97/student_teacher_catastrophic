from abc import ABC, abstractmethod

import torch

from typing import Dict, List, Union, overload

from models.networks.teachers import ClassificationTeacher, \
    RegressionTeacher, TrainedClassificationTeacher, _Teacher
from utils import Parameters


class _BaseTeachers(ABC):

    def __init__(self, config: Parameters):

        self._num_teachers = config.get(["task", "num_teachers"])
        self._device = config.get("device")
        self._teacher_configuration = \
            config.get(["task", "teacher_configuration"])

        self._loss_type = config.get(["task", "loss_type"])

        self._setup_teachers(config=config)

        self._teachers: List[_Teacher]

    @abstractmethod
    def _setup_teachers(self, config: Parameters) -> None:
        """instantiate teacher network(s)"""
        raise NotImplementedError("Base class method")

    def _init_teacher(self, config: Parameters, index: int):
        if self._loss_type == "classification":
            if self._teacher_configuration == "trained_mnist":
                return TrainedClassificationTeacher(
                    config=config, index=index
                    ).to(self._device)
            else:
                return ClassificationTeacher(
                    config=config, index=index
                    ).to(self._device)
        elif self._loss_type == "regression":
            return RegressionTeacher(
                config=config, index=index
                ).to(self._device)

    @overload
    @abstractmethod
    def test_set_forward(
        self,
        batch: List[Dict[str, torch.Tensor]]
    ) -> List[torch.Tensor]: ...

    @overload
    @abstractmethod
    def test_set_forward(
        self,
        batch: Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
    ) -> List[torch.Tensor]: ...

    @abstractmethod
    def test_set_forward(self, batch) -> List[torch.Tensor]:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def forward(
        self,
        teacher_index: int,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """call to current teacher forward"""
        raise NotImplementedError("Base class method")

    def forward_all(self, batch: Dict) -> List[torch.Tensor]:
        """call to forward of all teachers (used primarily for evaluation)"""
        outputs = [self.forward(t, batch) for t in range(self._num_teachers)]
        return outputs

    def get_teacher_networks(self) -> List[_Teacher]:
        """getter method for teacher networks"""
        networks = self._teachers
        return networks

    @abstractmethod
    def signal_task_boundary_to_teacher(self, new_task: int) -> None:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def signal_step_boundary_to_teacher(
        self,
        step: int,
        current_task: int
    ) -> None:
        raise NotImplementedError("Base class method")
