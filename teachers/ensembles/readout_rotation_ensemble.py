import copy
from typing import Dict
from typing import List

import numpy as np
import torch
from teachers.ensembles import base_teacher_ensemble
from utils import custom_functions


class ReadoutRotationTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble in which features (input to hidden) weights are copied from teacher
    to teacher by specified amount and readout weights (hidden to output) are rotated."""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        scale_hidden_lr: bool,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        normalise_teachers: bool,
        num_teachers: int,
        initialisation_std: float,
        feature_copy_percentage: int,
        rotation_magnitude: float,
    ):
        self._feature_copy_percentage = feature_copy_percentage
        self._rotation_magnitude = rotation_magnitude
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            scale_hidden_lr=scale_hidden_lr,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            num_teachers=num_teachers,
            initialisation_std=initialisation_std,
        )

    def _setup_teachers(self) -> None:
        """Setup teachers with copies across input to hidden and rotations
        across hidden to output weights.

        Raises:
            AssertionError: If more than 2 teachers are requested.
            AssertionError: If the network depth is greater than 1,
            i.e. more than one hidden layer requested.
            AssertionError: If the hidden dimension is not greater than 1,
            this is for the notion of rotation to have meaning.
        """

        assert (
            self._num_teachers
        ) == 2, "Readout rotation teachers currently implemented for 2 teachers only."

        assert (
            len(self._hidden_dimensions) == 1
        ), "Readout rotation teachers currently implemented for 1 hidden layer only."

        assert (
            self._hidden_dimensions[0] > 1
        ), "Readout rotation teachers only valid for hidden dimensions > 1."

        teachers = [self._init_teacher() for _ in range(self._num_teachers)]

        with torch.no_grad():
            # copy specified fraction of first teacher input -> hidden
            # weights to second teacher.
            index_weight_copy = int(
                self._input_dimension * self._feature_copy_percentage / 100
            )
            for h in range(self._hidden_dimensions[0]):
                teacher_0_weight_vector = teachers[0].layers[0].weight.data[h]
                teachers[1].layers[0].weight.data[h][
                    :index_weight_copy
                ] = teacher_0_weight_vector[:index_weight_copy]

            # rotate hidden -> output weights by specified amount.
            rotated_weight_vectors = custom_functions.generate_rotated_vectors(
                dimension=self._hidden_dimensions[0],
                theta=self._rotation_magnitude,
                normalisation=1,
            )

            teacher_0_rotated_weight_tensor = torch.Tensor(
                rotated_weight_vectors[0]
            ).reshape(teachers[0].head.weight.data.shape)

            teacher_1_rotated_weight_tensor = torch.Tensor(
                rotated_weight_vectors[1]
            ).reshape(teachers[1].head.weight.data.shape)

            teachers[0].head.weight.data = teacher_0_rotated_weight_tensor
            teachers[1].head.weight.data = teacher_1_rotated_weight_tensor

        return teachers
