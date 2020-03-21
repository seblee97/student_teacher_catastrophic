from .base_teacher import _Teacher

from typing import Dict 
import torch

class RegressionTeacher(_Teacher):

    def __init__(self, config: Dict, index: int) -> None:

        _Teacher.__init__(self, config=config, index=index)

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = self.output_layer(x)
        if self.noisy:
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        
        return y
