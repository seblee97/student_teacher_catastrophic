from .base_data_module import _BaseData
from utils import Parameters
from constants import Constants

import torch
import torch.distributions as tdist

from typing import Dict


class IIDData(_BaseData):
    """
    Class for generating data drawn i.i.d from unit normal 
    Gaussian.
    """
    def __init__(self, config: Parameters):
        _BaseData.__init__(self, config)

        mean = config.get(["iid_data", "mean"])
        variance = config.get(["iid_data", "variance"])

        self._data_distribution = tdist.Normal(mean, variance)

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        """
        This method gives a fixed test data set (input data only)

        Returns:
            test_input_batch: Dictionary with input data only
        """
        test_input_data = self._data_distribution.sample(
            (self._test_batch_size, self._input_dimension)
            ).to(self._device)

        test_data_dict = {'x': test_input_data}

        return test_data_dict

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """returns batch of training data (input only)"""
        batch = self._data_distribution.sample(
            (self._train_batch_size, self._input_dimension)
            ).to(self._device)
        return {'x': batch}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        pass
