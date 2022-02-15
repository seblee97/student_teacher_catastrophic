from typing import Callable
from typing import List
from typing import Union

import torch
from cata import constants
from cata.teachers import base_teacher
from cata.utils import threshold_functions


class ClassificationTeacher(base_teacher.BaseTeacher):
    """Classification - threshold output"""

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
        zero_head: bool = False
    ):
        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=constants.CLASSIFICATION,
            nonlinearity=nonlinearity,
            forward_hidden_scaling=forward_hidden_scaling,
            forward_scaling=forward_scaling,
            unit_norm_teacher_head=unit_norm_teacher_head,
            weight_normalisation=weight_normalisation,
            noise_std=noise_std,
            initialisation_std=initialisation_std,
        )
        self._threshold_fn = self._setup_threshold()

    def _setup_threshold(self) -> Callable:
        # threshold differently depending on nonlinearity to
        # ensure even class distributions
        if self._nonlinearity == constants.RELU:
            threshold_fn = threshold_functions.positive_threshold
        elif self._nonlinearity == constants.LINEAR:
            threshold_fn = threshold_functions.tanh_threshold
        elif self._nonlinearity == constants.SCALED_ERF:
            threshold_fn = threshold_functions.tanh_threshold
        else:
            raise NotImplementedError(
                f"Teacher thresholding for {self._nonlinearity}"
                " nonlinearity not yet implemented"
            )

        return threshold_fn

    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through head."""
        y = self._head(x)
        thresholded_output = self._threshold_fn(y)
        return thresholded_output
