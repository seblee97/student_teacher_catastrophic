from abc import ABC
from abc import abstractmethod
from typing import Dict

import torch

from constants import Constants
from utils import Parameters


class _BaseData(ABC):
    """
    Class for handling data

    Abstract methods that need to be implemented:

    - get_test_data
    - get_batch
    - signal_task_bounary_to_data_generator
    """

    def __init__(self, config: Parameters):
        """Init, extract relevant parameters from confg"""
        self._num_teachers: int = config.get(["task", "num_teachers"])

        self._train_batch_size: int = \
            config.get(["training", "train_batch_size"])
        self._test_batch_size: int = config.get(["testing", "test_batch_size"])
        self._input_dimension: int = config.get(["model", "input_dimension"])

        self._device = config.get("device")

    @abstractmethod
    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        """returns fixed test data sets (data and labels)"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """returns batch of training data (input data and label if relevant)"""
        raise NotImplementedError("Base class method")

    @abstractmethod
    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        """for use in cases where data generation changes with task
        (e.g. pure MNIST)
        """
        raise NotImplementedError("Base class method")
