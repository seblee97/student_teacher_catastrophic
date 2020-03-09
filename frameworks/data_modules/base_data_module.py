from abc import ABC, abstractmethod

from typing import Dict

import torch

class _BaseData(ABC):

    """Class for handling data"""

    def __init__(self, config: Dict):
        
        self._train_batch_size = config.get(["training", "train_batch_size"])
        self._test_batch_size = config.get(["training", "test_batch_size"])
        self._input_dimension = config.get(["model", "input_dimension"])

        self._device = config.get("device")
    
    @abstractmethod
    def get_test_set(self) -> List[torch.Tensor, torch.Tensor]:
        """returns fixed test data set (data and labels)"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def get_batch(self) -> torch.Tensor:
        """returns batch of training data"""
        raise NotImplementedError("Base class method")
                