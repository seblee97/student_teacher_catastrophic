import abc
import constants

import torch

from typing import Dict
from typing import List

from teachers import classification_teacher
from teachers import regression_teacher


class BaseTeacherEnsemble(abc.ABC):
    """Base class for sets/ensembles of teachers
    (as opposed to single teacher network)."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        unit_norm_teacher_head: bool,
        num_teachers: int,
        initialisation_std: float,
    ) -> None:
        self._input_dimension = input_dimension
        self._hidden_dimensions = hidden_dimensions
        self._output_dimension = output_dimension
        self._bias = bias
        self._loss_type = loss_type
        self._nonlinearity = nonlinearity
        self._unit_norm_teacher_head = unit_norm_teacher_head
        self._num_teachers = num_teachers
        self._initialisation_std = initialisation_std

        self._setup_teachers()

    @abc.abstractmethod
    def _setup_teachers(self) -> None:
        """instantiate teacher network(s)"""
        pass

    @abc.abstractmethod
    def test_set_forward(self, batch) -> List[torch.Tensor]:
        pass

    @abc.abstractmethod
    def forward(
        self, teacher_index: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Call to current teacher forward."""
        pass

    def _init_teacher(self):
        if self._loss_type == constants.Constants.CLASSIFICATION:
            teacher = classification_teacher.ClassificationTeacher
        elif self._loss_type == constants.Constants.REGRESSION:
            teacher = regression_teacher.RegressionTeacher
        else:
            raise ValueError(f"Loss type {self._loss_type} is not recognised.")
        return teacher(
            input_dimension=self._input_dimension,
            hidden_dimensions=self._hidden_dimensions,
            output_dimension=self._output_dimension,
            bias=self._bias,
            nonlinearity=self._nonlinearity,
            unit_norm_teacher_head=self._unit_norm_teacher_head,
            initialisation_std=self._initialisation_std,
        )

    def save_weights(self, teacher_index: int, save_path: str):
        """Save weights associated with given teacher index"""
        torch.save(self._teachers[teacher_index].state_dict(), save_path)

    def forward_all(self, batch: Dict) -> List[torch.Tensor]:
        """Call to forward of all teachers (used primarily for evaluation)"""
        outputs = [self.forward(t, batch) for t in range(self._num_teachers)]
        return outputs

    @property
    def teachers(self) -> List:
        """Getter method for teacher networks."""
        return self._teachers
