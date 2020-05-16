from .mnist_data import _MNISTData
from utils import Parameters
from constants import Constants

from typing import Dict, Tuple, List

import torch
import torchvision
from torch.utils.data import Dataset


class MNISTEvenGreaterData(_MNISTData):

    def __init__(self, config: Parameters):

        self._current_teacher_index: int
        self._num_teachers: int = config.get(["task", "num_teachers"])

        _MNISTData.__init__(self, config)

        self.train_data: List[Dataset]
        self.test_data: List[Dataset]

        self.training_data_iterators = [
            self._generate_iterator(
                dataset=dataset, batch_size=self._train_batch_size,
                shuffle=True
            )
            for dataset in self.train_data
        ]

        self.test_data_iterators = [
            self._generate_iterator(
                dataset=dataset, batch_size=None,
                shuffle=False
            )
            for dataset in self.test_data
        ]

    def _generate_datasets(
        self
    ) -> Tuple[Constants.DATASET_TYPES, Constants.DATASET_TYPES]:
        """
        """
        # first dataset
        even_odd_train_dataset = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=True,
                target_transform=lambda x: Constants.EVEN_ODD_MAPPING[x]
                )
        even_odd_test_dataset = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=False,
                target_transform=lambda x: Constants.EVEN_ODD_MAPPING[x]
                )

        greater_five_train_dataset = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=True,
                target_transform=lambda x: Constants.GREATER_FIVE_MAPPING[x]
                )
        greater_five_test_dataset = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=False,
                target_transform=lambda x: Constants.GREATER_FIVE_MAPPING[x]
                )

        train_data = [
            even_odd_train_dataset, greater_five_train_dataset
            ]
        test_data = [
            even_odd_test_dataset, greater_five_test_dataset
            ]

        return train_data, test_data

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        """
        """
        full_test_datasets: List[torch.Tensor] = [
            next(test_set_iterator)
            for test_set_iterator in self.test_data_iterators
            ]

        assert torch.equal(
                full_test_datasets[0][0], full_test_datasets[1][0]
            ), (
                "Inputs for test set (and training set) should be identical"
                "for both tasks"
            )

        # arbitrarily have one of two dataset inputs as test set inputs
        input_data = full_test_datasets[0][0]
        # labels obviously still differ between tasks

        labels = [test_set[1].unsqueeze(1) for test_set in full_test_datasets]

        return {'x': input_data, 'y': labels}

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        returns batch of training data. Retrieves next batch from dataloader
        iterator.
        If iterator is empty, it is reset. Returns both input and label.

        return batch_input:
        """
        try:
            batch = next(
                self.training_data_iterators[self._current_teacher_index]
                )
        except StopIteration:
            dataset = self.train_data[self._current_teacher_index]
            self.training_data_iterators[self._current_teacher_index] = \
                self._generate_iterator(
                    dataset=dataset, batch_size=self._train_batch_size,
                    shuffle=True
                )
            batch = \
                next(self.training_data_iterators[self._current_teacher_index])

        y = batch[1].reshape((self._train_batch_size, -1)).type(
            torch.FloatTensor
            )

        return {'x': batch[0], 'y': y}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        self._current_teacher_index = new_task
