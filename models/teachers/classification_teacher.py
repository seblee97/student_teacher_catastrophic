from .base_teacher import _Teacher

from typing import Dict
import torch

class ClassificationTeacher(_Teacher):

    """Classification - threshold output"""

    def __init__(self, config: Dict, index: int) -> None:

        _Teacher.__init__(self, config=config, index=index)

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override teacher output forward, add sign output
        """
        y = self.output_layer(x)
        if self.noisy:
            noise = self.noise_distribution.sample((y.shape[0],))
            y = y + noise
        
        # threshold
        tanh_y = torch.tanh(y)
        return (torch.abs(tanh_y) > 0.5).type(torch.LongTensor).reshape(len(tanh_y),)