import abc
import math
from typing import Dict
from typing import List

import torch

import constants
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
        scale_hidden_lr: bool,
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

        if scale_hidden_lr:
            self._forward_scaling = 1 / math.sqrt(input_dimension)
        else:
            self._forward_scaling = 1.0

        self._teachers = self._setup_teachers()

    @property
    def teachers(self) -> List:
        """Getter method for teacher networks."""
        return self._teachers

    @property
    def cross_overlaps(self):
        overlaps = []
        for i in range(len(self._teachers)):
            for j in range(i, len(self._teachers)):
                if i != j:
                    overlap = (
                        torch.mm(
                            self._teachers[i].layers[0].weight.data,
                            self._teachers[j].layers[0].weight.data.T,
                        )
                        / self._input_dimension
                    )
                    overlaps.append(overlap)
        return overlaps

    @abc.abstractmethod
    def _setup_teachers(self) -> None:
        """instantiate teacher network(s)"""
        pass

    def forward(
        self, teacher_index: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Call to current teacher forward."""
        x = batch[constants.Constants.X]
        output = self._teachers[teacher_index](x)
        return output

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
            forward_scaling=self._forward_scaling,
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
