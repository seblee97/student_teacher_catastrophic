from .base_teacher import _Teacher
from utils import Parameters

from typing import Dict 
import torch

class RegressionTeacher(_Teacher):

    def __init__(self, config: Parameters, index: int) -> None:

        _Teacher.__init__(self, config=config, index=index)

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = self.output_layer(x)

        if self.noisy:
            y = self.add_output_noise(y)

        return y
