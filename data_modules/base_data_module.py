import abc
from typing import Dict

import torch


class BaseData(abc.ABC):
    """Class for handling data

    Abstract methods that need to be implemented:

    - get_test_data
    - get_batch
    - signal_task_bounary_to_data_generator
    """

    def __init__(
        self, train_batch_size: int, test_batch_size: int, input_dimension: int
    ):
        """Class constructor"""
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        self._input_dimension = input_dimension

    @abc.abstractmethod
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """returns fixed test data sets (data and labels)"""
        raise NotImplementedError("Base class method")

    @abc.abstractmethod
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """returns batch of training data (input data and label if relevant)"""
        raise NotImplementedError("Base class method")
