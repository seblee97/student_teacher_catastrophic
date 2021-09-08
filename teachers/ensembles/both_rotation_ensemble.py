import copy
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import torch
from teachers.ensembles import base_teacher_ensemble
from utils import custom_functions


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
        nonlinearities: str,
        scale_hidden_lr: bool,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        noise_stds: Union[int, float],
        num_teachers: int,
        initialisation_std: float,
        feature_rotation_alpha: float,
        readout_rotation_alpha: float,
    ):
        self._feature_rotation_alpha = feature_rotation_alpha
        self._readout_rotation_alpha = readout_rotation_alpha
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

        teachers = [
            self._init_teacher(
                nonlinearity=self._nonlinearities[i], noise_std=self._noise_stds[i]
            )
            for i in range(self._num_teachers)
        ]

        with torch.no_grad():

            (
                teacher_0_feature_weights,
                teacher_1_feature_weights,
            ) = self._get_rotated_weights(
                unrotated_weights=teachers[0].layers[0].weight.data.T,
                alpha=self._feature_rotation_alpha,
                normalisation=self._hidden_dimensions[0],
            )

            teachers[0].layers[0].weight.data = teacher_0_feature_weights.T
            teachers[1].layers[0].weight.data = teacher_1_feature_weights.T

            # (
            #     teacher_0_readout_weights,
            #     teacher_1_readout_weights,
            # ) = self._get_rotated_weights(
            #     unrotated_weights=teachers[0].head.weight.data.T,
            #     alpha=self._readout_rotation_alpha,
            #     normalisation=None,
            # )

            (
                teacher_0_readout_weights,
                teacher_1_readout_weights,
            ) = self._get_rotated_readout_weights(teachers=teachers)

            teachers[0].head.weight.data = teacher_0_readout_weights
            teachers[1].head.weight.data = teacher_1_readout_weights

        return teachers

    def _feature_overlap(self, feature_1: torch.Tensor, feature_2: torch.Tensor):
        alpha_matrix = torch.mm(feature_1, feature_2.T) / self._hidden_dimensions[0]
        alpha = torch.mean(alpha_matrix.diagonal())

        return alpha

    def _readout_overlap(self, feature_1: torch.Tensor, feature_2: torch.Tensor):
        alpha = torch.mm(feature_1, feature_2.T) / (
            torch.norm(feature_1) * torch.norm(feature_2)
        )
        return alpha

    def _get_rotated_weights(
        self,
        unrotated_weights: torch.Tensor,
        alpha: float,
        normalisation: Union[None, int],
    ):
        if normalisation is not None:
            # orthonormalise input to hidden weights of first teacher
            self_overlap = (
                torch.mm(unrotated_weights, unrotated_weights.T) / normalisation
            )
            L = torch.cholesky(self_overlap)
            orthonormal_weights = torch.mm(torch.inverse(L), unrotated_weights)
        else:
            orthonormal_weights = unrotated_weights

        # construct input to hidden weights of second teacher
        second_teacher_rotated_weights = alpha * orthonormal_weights + np.sqrt(
            1 - alpha ** 2
        ) * torch.randn(orthonormal_weights.shape)

        return orthonormal_weights, second_teacher_rotated_weights

    def _get_rotated_readout_weights(self, teachers: List):

        theta = np.arccos(self._readout_rotation_alpha)

        # keep current norms
        current_norm = np.mean(
            [torch.norm(teacher.head.weight) for teacher in teachers]
        )

        rotated_weight_vectors = custom_functions.generate_rotated_vectors(
            dimension=self._hidden_dimensions[0],
            theta=theta,
            normalisation=current_norm,
        )

        teacher_0_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[0]
        ).reshape(teachers[0].head.weight.data.shape)

        teacher_1_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[1]
        ).reshape(teachers[1].head.weight.data.shape)

        return teacher_0_rotated_weight_tensor, teacher_1_rotated_weight_tensor
