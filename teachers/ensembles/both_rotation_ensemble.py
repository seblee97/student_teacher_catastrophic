import copy
from typing import Dict
from typing import List

import numpy as np
import torch

from utils import custom_functions
from teachers.ensembles import base_teacher_ensemble


class BothRotationTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble (primarily for mean-field limit regime) in which both feature and
    readout similarities are tuned by rotation.
    """

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
        num_teachers: int,
        initialisation_std: float,
        feature_rotation_alpha: float,
        readout_rotation_magnitude: float,
    ):
        self._feature_rotation_alpha = feature_rotation_alpha
        self._readout_rotation_magnitude = readout_rotation_magnitude
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
        ) == 2, "Both rotation teachers currently implemented for 2 teachers only."

        assert (
            len(self._hidden_dimensions) == 1
        ), "Both rotation teachers currently implemented for 1 hidden layer only."

        assert (
            self._hidden_dimensions[0] > 1
        ), "Both rotation teachers only valid for hidden dimensions > 1."

        teachers = [self._init_teacher() for _ in range(self._num_teachers)]

        with torch.no_grad():
            # orthonormalise input to hidden weights of first teacher
            first_teacher_feature_weights = teachers[0].layers[0].weight.data.T
            self_overlap = (
                torch.mm(first_teacher_feature_weights, first_teacher_feature_weights.T)
                / self._hidden_dimensions[0]
            )
            L = torch.cholesky(self_overlap)
            orthonormal_weights = torch.mm(
                torch.inverse(L), first_teacher_feature_weights
            )

            # construct input to hidden weights of second teacher
            second_teacher_rotated_weights = (
                self._feature_rotation_alpha * orthonormal_weights
                + np.sqrt(1 - self._feature_rotation_alpha ** 2)
                * torch.randn(orthonormal_weights.shape)
            )

            teachers[0].layers[0].weight.data = orthonormal_weights.T
            teachers[1].layers[0].weight.data = second_teacher_rotated_weights.T

            # rotate hidden -> output weights by specified amount.

            # keep current norms
            current_norm = np.mean(
                [torch.norm(teacher.head.weight) for teacher in teachers]
            )

            rotated_weight_vectors = custom_functions.generate_rotated_vectors(
                dimension=self._hidden_dimensions[0],
                theta=self._readout_rotation_magnitude,
                normalisation=current_norm,
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
