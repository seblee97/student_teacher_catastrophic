from typing import List

import constants
import torch
from teachers import base_teacher


class RegressionTeacher(base_teacher.BaseTeacher):
    """Regression - no threshold on output"""

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        nonlinearity: str,
        forward_hidden_scaling: float,
        forward_scaling: float,
        unit_norm_teacher_head: bool,
        weight_normalisation: bool,
        initialisation_std: float,
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=constants.Constants.REGRESSION,
            nonlinearity=nonlinearity,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            initialisation_std=initialisation_std,
        )

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head."""
        y = self._head(x)
        return y
