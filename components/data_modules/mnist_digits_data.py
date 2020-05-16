from .mnist_data import _MNISTData
from utils import Parameters
from constants import Constants

from typing import Dict, Tuple, List

import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, Dataset

import numpy as np
import copy


class MNISTDigitsData(_MNISTData):

    def __init__(self, config: Parameters, override_batch_size: int = None):

        self.override_batch_size = override_batch_size

        self._mnist_teacher_classes = \
            config.get(['pure_mnist', 'teacher_digits'])

        _MNISTData.__init__(self, config)

        if self._num_teachers != len(self._mnist_teacher_classes):
            raise AssertionError(
                "Number of teachers specified in base config is not \
                    compatible with number of MNIST classifiers specified."
                )

        if self.override_batch_size is not None:
            self._batch_size = self.override_batch_size
        else:
            self._batch_size = self._train_batch_size

        self._current_teacher_index: int
        self.train_data: List[Dataset]
        self.test_data: List[Dataset]

        self.training_data_iterators = [
                self._generate_iterator(
                    dataset=dataset, batch_size=self._batch_size,
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

    def filter_dataset_by_target(
        self,
        unflitered_data: Constants.MNIST_DATASET_TYPE,
        target_to_keep: int,
        train: bool,
        new_label: int = None
    ) -> Subset:
        """generate subset of dataset filtered by target"""
        targets = unflitered_data.targets

        indices_to_keep = targets == target_to_keep
        filtered_dataset_indices = np.arange(len(targets))[indices_to_keep]

        if train:
            train_dataset_copy = copy.deepcopy(unflitered_data)
            for ind in filtered_dataset_indices:
                train_dataset_copy.targets[ind] = torch.tensor([new_label])
            return Subset(train_dataset_copy, filtered_dataset_indices)
        else:
            return Subset(unflitered_data, filtered_dataset_indices)

    def _generate_datasets(
        self
    ) -> Tuple[Constants.DATASET_TYPES, Constants.DATASET_TYPES]:

        full_train_data = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=True
                )
        full_test_data = \
            torchvision.datasets.MNIST(
                self._full_data_path, transform=self.transform, train=False
                )

        train_datasets = []
        test_datasets = []

        for t, task_classes in enumerate(self._mnist_teacher_classes):

            # get filtered training sets
            train_target_filtered_datasets = [
                self.filter_dataset_by_target(
                    full_train_data, target, train=True, new_label=st
                    )
                for st, target in enumerate(task_classes)
                ]
            train_task_dataset: ConcatDataset = \
                ConcatDataset(train_target_filtered_datasets)

            # get filtered test sets
            test_target_filtered_datasets = [
                self.filter_dataset_by_target(
                    full_test_data, target, train=False
                    )
                for target in task_classes
                ]
            test_task_dataset: ConcatDataset = \
                ConcatDataset(test_target_filtered_datasets)

            train_datasets.append(train_task_dataset)
            test_datasets.append(test_task_dataset)

        return train_datasets, test_datasets

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:

        full_test_datasets = [
            next(test_set_iterator)[0]
            for test_set_iterator in self.test_data_iterators
            ]

        data = torch.cat([test_set[0] for test_set in full_test_datasets])
        labels = [test_set[1] for test_set in full_test_datasets]
        return {'x': data, 'y': labels}

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        returns batch of training data. Retrieves next batch from dataloader
        iterator.
        If iterator is empty, it is reset. Returns both input and label.

        return batch_input:
        """
        try:
            batch = \
                next(self.training_data_iterators[self._current_teacher_index])
        except StopIteration:
            dataset = self.train_data[self._current_teacher_index]
            self.training_data_iterators[self._current_teacher_index] = \
                self._generate_iterator(
                    dataset=dataset, batch_size=self._batch_size,
                    shuffle=True
                )
            batch = \
                next(self.training_data_iterators[self._current_teacher_index])

        return {'x': batch[0], 'y': batch[1]}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        self._current_teacher_index = new_task
