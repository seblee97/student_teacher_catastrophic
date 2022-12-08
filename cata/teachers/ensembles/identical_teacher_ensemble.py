from typing import List
from typing import Union

import numpy as np
import torch

from cata.teachers.ensembles import base_teacher_ensemble
from cata.utils import custom_functions


class IdenticalTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble in which teachers are identical."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearities: str,
        scale_hidden_lr: bool,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_stds: Union[int, float],
        num_teachers: int,
        initialisation_std: float,
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearities=nonlinearities,
            scale_hidden_lr=scale_hidden_lr,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            noise_stds=noise_stds,
            num_teachers=num_teachers,
            initialisation_std=initialisation_std,
        )

    def _setup_teachers(self) -> None:
        """Instantiate teachers with input to hidden weights rotated."""

        assert (
            self._num_teachers
        ) == 2, "Feature rotation teachers currently implemented for 2 teachers only."

        teachers = [
            self._init_teacher(
                nonlinearity=self._nonlinearities[i],
                noise_std=self._noise_stds[i],
                zero_head=0,
            )
            for i in range(self._num_teachers)
        ]

        input_hidden_weights_to_be_copied = (
            teachers[0].layers[0].weight.data
        )

        hidden_output_weights_to_be_copied = (
            teachers[0].head.weight.data
        )

        with torch.no_grad():

            for i in range(self._num_teachers):

                teachers[i].layers[0].weight.data = input_hidden_weights_to_be_copied
                teachers[i].head.weight.data = hidden_output_weights_to_be_copied

        return teachers
