from typing import Dict
from typing import Tuple

import torch
import torchvision
from torch.utils.data import Dataset

from constants import Constants
from utils import Parameters
from .mnist_data import _MNISTData


class MNISTStreamData(_MNISTData):
    """
    Data class for the setting of MNIST digits being flattened
    and fed as inputs to a student-teacher framework.

    This class inherits from _MNISTData, which in turn
    inherits from the base data class, _BaseData.

    This class implements the following abstract methods from
    _MNISTData:

    - _generate_datasets

    and the following abstract methods from _BaseData:

    - get_test_data
    - get_batch
    - signal_task_boundary_to_data_generator
    """

    def __init__(self, config: Parameters):
        """
        This method initialises the class. It loads the
        datasets and creates the training and test iterator.
        """
        _MNISTData.__init__(self, config)

        self.train_data: Dataset
        self.test_data: Dataset

        self.training_data_iterator = self._generate_iterator(
            dataset=self.train_data, batch_size=self._train_batch_size, shuffle=True)

        self.test_data_iterator = self._generate_iterator(
            dataset=self.test_data, batch_size=None, shuffle=False)

    def _generate_datasets(self) -> Tuple[Constants.DATASET_TYPES, Constants.DATASET_TYPES]:
        """
        This method loads the MNIST datasets
        """
        train_data = torchvision.datasets.MNIST(
            self._full_data_path, transform=self.transform, train=True)
        test_data = torchvision.datasets.MNIST(
            self._full_data_path, transform=self.transform, train=False)

        return train_data, test_data

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        """
        This method returns fixed test data set (input data only).

        Returns:
            - test_data_dict: Dictionary containing 'x', the input
            test data only.
        """
        data, labels = next(self.test_data_iterator)

        test_data_dict = {'x': data}

        return test_data_dict

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        This method returns batch of training data (input and labels).
        Retrieves next batch from dataloader iterator.

        If iterator is empty, it is reset.

        Returns:
            batch: Dictionary containing 'x', input
            of training data batch
        """
        try:
            batch_input = next(self.training_data_iterator)[0]
        except StopIteration:
            self.training_data_iterator = \
                self._generate_iterator(
                    dataset=self.train_data,
                    batch_size=self._train_batch_size,
                    shuffle=True
                )
            batch_input = next(self.training_data_iterator)[0]

        return {'x': batch_input}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        pass
