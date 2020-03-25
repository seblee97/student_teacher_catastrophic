from models.base_network import Model

import torch.nn as nn
import torch

import numpy as np

from typing import Dict, List

from abc import ABC, abstractmethod

class _BaseStudent(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')
        
        if config.get(["task", "loss_type"]) == "classification":
            self.classification_output = True
        elif config.get(["task", "loss_type"]) == 'regression':
            self.classification_output = False
        else:
            raise ValueError("Unknown loss type given in base config")
    
    @abstractmethod
    def _construct_output_layers(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def set_task(self, task_index: int):
        raise NotImplementedError("Base class method")

    @abstractmethod    
    def forward_all(self, x: torch.Tensor):
        raise NotImplementedError("Base class method")
    
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._get_output_from_head(x)

        # threshold 
        if self.classification_output:
            return self._threshold(y)
            
        return y

    def _threshold(self, y):
        return torch.sigmoid(y)
    
    @abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _get_head_weights(self) -> List[torch.Tensor]:
        raise NotImplementedError("Base class method")
