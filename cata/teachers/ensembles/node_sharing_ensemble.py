from typing import List
from typing import Union

import numpy as np
import torch
from cata.teachers.ensembles import base_teacher_ensemble
from cata.utils import custom_functions


class NodeSharingTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble in which a subset of nodes are shared between teachers
    and the reamining features (input to hidden) weights are rotated by
    a given amount."""

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
        num_shared_nodes: int,
        feature_rotation_magnitude: float,
    ):
        self._num_shared_nodes = num_shared_nodes
        self._rotation_magnitude = feature_rotation_magnitude
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

        assert (
            len(self._hidden_dimensions) == 1
        ), "Feature rotation teachers currently implemented for 1 hidden layer only."

        teachers = [
            self._init_teacher(
                nonlinearity=self._nonlinearities[i],
                noise_std=self._noise_stds[i],
                zero_head=0,
            )
            for i in range(self._num_teachers)
        ]

        weights_to_be_copied = (
            teachers[0].layers[0].weight.data[: self._num_shared_nodes]
        )
        weights_to_be_rotated = (
            teachers[0].layers[0].weight.data[self._num_shared_nodes :]
        )

        with torch.no_grad():

            if self._hidden_dimensions[0] - self._num_shared_nodes == 1:
                rotated_weight_vectors = custom_functions.generate_rotated_vectors(
                    dimension=self._input_dimension,
                    theta=self._rotation_magnitude,
                    normalisation=np.sqrt(self._input_dimension),
                )
            else:
                rotated_weight_vectors = custom_functions.generate_rotated_matrices(
                    unrotated_weights=weights_to_be_rotated,
                    alpha=np.cos(self._rotation_magnitude),
                    normalisation=self._input_dimension,
                    orthogonalise=False,
                )

            for i in range(self._num_teachers):
                teacher_weight_tensors = torch.cat(
                    (
                        weights_to_be_copied,
                        torch.Tensor(
                            rotated_weight_vectors[i].reshape(
                                self._hidden_dimensions[0] - self._num_shared_nodes,
                                self._input_dimension,
                            )
                        ),
                    ),
                    dim=0,
                )

                teachers[i].layers[0].weight.data = teacher_weight_tensors

        return teachers
