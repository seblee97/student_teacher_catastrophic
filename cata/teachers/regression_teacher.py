from typing import List
from typing import Union

import torch
from cata import constants
from cata.teachers import base_teacher


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
        noise_std: Union[float, int],
        initialisation_std: float,
        zero_head: bool = False,
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=constants.REGRESSION,
            nonlinearity=nonlinearity,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            noise_std=noise_std,
            initialisation_std=initialisation_std,
            zero_head=zero_head,
        )
        if self._noise_std != 0:
        #print(f"{self._noise_std}")
            self._noise_module = torch.distributions.normal.Normal(
                loc=0, scale=self._noise_std
            )
        else:
            print("NOISELESS")

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head."""
        noiseless_y = self._head(x)
        if self._noise_std == 0:
            noise = 0
        else:
            noise = self._noise_module.sample(noiseless_y.shape)
            
        y = noiseless_y + noise
        return y
