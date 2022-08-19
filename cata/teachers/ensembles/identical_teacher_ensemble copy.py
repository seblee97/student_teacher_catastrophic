import abc
import math
from typing import List
from typing import Union

import torch
from cata import constants
from cata.teachers import classification_teacher
from cata.teachers import regression_teacher


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
        nonlinearities: List[str],
        scale_hidden_lr: bool,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_stds: List[Union[int, float]],
        num_teachers: int,
        initialisation_std: float,
    ) -> None:
        self._input_dimension = input_dimension
        self._hidden_dimensions = hidden_dimensions
        self._output_dimension = output_dimension
        self._bias = bias
        self._loss_type = loss_type
        self._nonlinearities = nonlinearities
        self._forward_scaling = forward_scaling
        self._unit_norm_teacher_head = unit_norm_teacher_head
        self._weight_normalisation = weight_normalisation
        self._noise_stds = noise_stds
        self._num_teachers = num_teachers
        self._initialisation_std = initialisation_std

        if scale_hidden_lr:
            self._forward_hidden_scaling = 1 / math.sqrt(input_dimension)
        else:
            self._forward_hidden_scaling = 1.0

        self._teachers = self._setup_teachers()

    @property
    def teachers(self) -> List:
        """Getter method for teacher networks."""
        return self._teachers

    @property
    def cross_overlaps(self):
        overlaps = []
        with torch.no_grad():
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
        #why is numpy not recognised?
        dimension=self._input_dimension,
        normalisation=np.sqrt(self._input_dimension),
        v_1 = np.random.normal(size=(dimension))
        v_2 = np.random.normal(size=(dimension))
        normal_1 = normalisation * v_1 / np.linalg.norm(v_1)
        normal_2 = normalisation * v_2 / np.linalg.norm(v_2)



    def forward(self, teacher_index: int, batch: torch.Tensor) -> torch.Tensor:
        """Call to current teacher forward."""
        output = self._teachers[teacher_index](batch)
        return output

    def _init_teacher(
        self, nonlinearity: str, noise_std: Union[float, int], zero_head: bool = False
    ):
        if self._loss_type == constants.CLASSIFICATION:
            teacher = classification_teacher.ClassificationTeacher
        elif self._loss_type == constants.REGRESSION:
            teacher = regression_teacher.RegressionTeacher
        else:
            raise ValueError(f"Loss type {self._loss_type} is not recognised.")
        return teacher(
            input_dimension=self._input_dimension,
            hidden_dimensions=self._hidden_dimensions,
            output_dimension=self._output_dimension,
            bias=self._bias,
            nonlinearity=nonlinearity,
            forward_hidden_scaling=self._forward_hidden_scaling,
            forward_scaling=self._forward_scaling,
            unit_norm_teacher_head=self._unit_norm_teacher_head,
            weight_normalisation=self._weight_normalisation,
            noise_std=noise_std,
            initialisation_std=self._initialisation_std,
            zero_head=zero_head,
        )

    def save_all_teacher_weights(self, save_path: str) -> None:
        """Save weights associated with each teacher

        Args:
            save_path: path to save weights, will be concatenated with
            _i where i is the index of the teacher.
        """
        for t, teacher in enumerate(self._teachers):
            torch.save(teacher.state_dict(), f"{save_path}_{t}")

    def save_weights(self, teacher_index: int, save_path: str) -> None:
        """Save weights associated with given teacher index"""
        torch.save(self._teachers[teacher_index].state_dict(), save_path)

    def forward_all(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Call to forward of all teachers (used primarily for evaluation)"""
        outputs = [self.forward(t, batch) for t in range(self._num_teachers)]
        return outputs
