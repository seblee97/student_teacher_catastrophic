import copy
from typing import Dict
from typing import List

import numpy as np
import torch

from utils import custom_functions
from teachers.ensembles import base_teacher_ensemble


class FeatureRotationTeacherEnsemble(base_teacher_ensemble.BaseTeacherEnsemble):
    """Teacher ensemble in which features (input to hidden) weights are rotated."""

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
        rotation_magnitude: float,
    ):
        self._rotation_magnitude = rotation_magnitude
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            unit_norm_teacher_head=unit_norm_teacher_head,
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

        assert (
            self._hidden_dimensions[0] == 1
        ), "Feature rotation teachers implemented for hidden dimension 1 only."

        self._teachers = [self._init_teacher() for _ in range(self._num_teachers)]

        rotated_weight_vectors = custom_functions.generate_rotated_vectors(
            dimension=self._input_dimension,
            theta=self._rotation_magnitude,
            normalisation=np.sqrt(self._input_dimension),
        )

        teacher_0_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[0]
        ).reshape(self._teachers[0].layers[0].weight.data.shape)

        teacher_1_rotated_weight_tensor = torch.Tensor(
            rotated_weight_vectors[1]
        ).reshape(self._teachers[1].layers[0].weight.data.shape)

        self._teachers[0].layers[0].weight.data = teacher_0_rotated_weight_tensor
        self._teachers[1].layers[0].weight.data = teacher_1_rotated_weight_tensor

    def test_set_forward(self, batch) -> List[torch.Tensor]:
        raise NotImplementedError

    def forward(
        self, teacher_index: int, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError
