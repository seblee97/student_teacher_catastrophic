import abc
import math
from typing import List
from typing import Union
import numpy as np
import torch
from cata import constants
from cata.teachers import classification_teacher
from cata.teachers import regression_teacher


class IdenticalTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble in which features (input to hidden) weights are identical."""

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

    def _setup_teachers(self) -> None:
        """instantiate teacher network(s) with identical weights"""
        
        assert (
            self._num_teachers
        ) == 2, "Identical teachers currently implemented for 2 teachers only."

        assert (
            len(self._hidden_dimensions) == 1
        ), "Identical teachers currently implemented for 1 hidden layer only."

        assert (
            self._hidden_dimensions[0] > 1
        ), "Identical teachers only valid for hidden dimensions > 1."

        teachers = [
            self._init_teacher(
                nonlinearity=self._nonlinearities[i],
                noise_std=self._noise_stds[i],
                zero_head=False,
            )
            for i in range(self._num_teachers)
        ]
        with torch.no_grad():
            dimension=self._input_dimension,
            normalisation=np.sqrt(self._input_dimension),
            v = np.random.normal(size=(dimension))
            #v_2 = np.random.normal(size=(dimension))
            normal = normalisation * v / np.linalg.norm(v)
            teacher_0_tensor = torch.Tensor(normal).reshape(teachers[0].layers[0].weight.data.shape)
            teacher_1_tensor = torch.Tensor(normal).reshape(teachers[1].layers[0].weight.data.shape)

            teachers[0].layers[0].weight.data = teacher_0_tensor
            teachers[1].layers[0].weight.data = teacher_1_tensor

        return teachers










