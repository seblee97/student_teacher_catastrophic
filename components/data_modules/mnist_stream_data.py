from .mnist_data import _MNISTData
from utils import Parameters
from constants import Constants

import torch
import torchvision
from torch.utils.data import Dataset

from typing import Dict, Tuple


class MNISTStreamData(_MNISTData):

    def __init__(self, config: Parameters):
        _MNISTData.__init__(self, config)

        self.train_data: Dataset
        self.test_data: Dataset

        self.training_data_iterator = self._generate_iterator(
            dataset=self.train_data, batch_size=self._train_batch_size,
            shuffle=True
        )

        self.test_data_iterator = self._generate_iterator(
            dataset=self.test_data, batch_size=None,
            shuffle=False
        )

    def _generate_datasets(
        self
    ) -> Tuple[Constants.DATASET_TYPES, Constants.DATASET_TYPES]:

        train_data = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=True
                )
        test_data = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=False
                )

        return train_data, test_data

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        data, labels = next(self.test_data_iterator)
        return {'x': data}

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        returns batch of training data. Retrieves next batch from
        dataloader iterator.
        If iterator is empty, it is reset.

        return batch_input:
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
