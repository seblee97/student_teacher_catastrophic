from .base_data_module import _BaseData
from utils import Parameters

import torch

from typing import List, Dict, Union

class IIDData(_BaseData):

    """Class for dealing with data generated from IID Gaussian"""

    def __init__(self, config: Parameters):
        _BaseData.__init__(self, config)

    def get_test_data(self) -> Union[List[Dict[str, torch.Tensor]], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        returns fixed test data set (data and labels)
        
        return test_input_batch: test data set
        """
        test_input_data = torch.randn(self._test_batch_size, self._input_dimension).to(self._device)

        return {'x': test_input_data}

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """returns batch of training data"""
        batch = torch.randn(self._train_batch_size, self._input_dimension).to(self._device)
        return {'x': batch}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        pass