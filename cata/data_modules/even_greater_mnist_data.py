from typing import Dict
from typing import List
from typing import Tuple

import torch
import torchvision
from cata.constants import Constants
from cata.data_modules.mnist_data import _MNISTData
from cata.utils import Parameters
from torch.utils.data import Dataset


class MNISTEvenGreaterData(_MNISTData):
    """
    Data class for the setting of two tasks where
    one task is binary classification between even and odd
    digits and the other is binary classification between
    digits greater than / less than 5.

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
        datasets and creates iterators for the train and test data
        """
        self._current_teacher_index: int
        self._num_teachers: int = config.get(["task", "num_teachers"])

        _MNISTData.__init__(self, config)

        self.train_data: List[Dataset]
        self.test_data: List[Dataset]

        self.training_data_iterators = [
            self._generate_iterator(
                dataset=dataset, batch_size=self._train_batch_size, shuffle=True
            )
            for dataset in self.train_data
        ]

        self.test_data_iterators = [
            self._generate_iterator(dataset=dataset, batch_size=None, shuffle=False)
            for dataset in self.test_data
        ]

    def _generate_datasets(
        self,
    ) -> Tuple[Constants.DATASET_TYPES, Constants.DATASET_TYPES]:
        """
        This method loads the MNIST datasets and maps the output labels
        to the relevant binary labels.
        """
        # first dataset
        even_odd_train_dataset = torchvision.datasets.MNIST(
            self._full_data_path,
            transform=self.transform,
            train=True,
            target_transform=lambda x: Constants.EVEN_ODD_MAPPING[x],
        )
        even_odd_test_dataset = torchvision.datasets.MNIST(
            self._full_data_path,
            transform=self.transform,
            train=False,
            target_transform=lambda x: Constants.EVEN_ODD_MAPPING[x],
        )

        greater_five_train_dataset = torchvision.datasets.MNIST(
            self._full_data_path,
            transform=self.transform,
            train=True,
            target_transform=lambda x: Constants.GREATER_FIVE_MAPPING[x],
        )
        greater_five_test_dataset = torchvision.datasets.MNIST(
            self._full_data_path,
            transform=self.transform,
            train=False,
            target_transform=lambda x: Constants.GREATER_FIVE_MAPPING[x],
        )

        train_data = [even_odd_train_dataset, greater_five_train_dataset]
        test_data = [even_odd_test_dataset, greater_five_test_dataset]

        return train_data, test_data

    def get_test_data(self) -> Constants.TEST_DATA_TYPES:
        """
        This method returns fixed test data sets (data and labels).
        In this task setting the input distribution is the same, so
        the input test data is the same for both tasks.

        Returns:
            - test_data_dict: Dictionary containing 'x', the input
            test data and 'y' a list of two sets of output labels, one
            for each task

        Raises:
            - AssertionError if the input data tensors do not match
            for the two tasks
        """
        full_test_datasets: List[torch.Tensor] = [
            next(test_set_iterator) for test_set_iterator in self.test_data_iterators
        ]

        assert torch.equal(full_test_datasets[0][0], full_test_datasets[1][0]), (
            "Inputs for test set (and training set) should be identical"
            "for both tasks"
        )

        # arbitrarily have one of two dataset inputs as test set inputs
        input_data = full_test_datasets[0][0]

        # labels obviously still differ between tasks
        labels = [test_set[1].unsqueeze(1) for test_set in full_test_datasets]

        test_data_dict = {"x": input_data, "y": labels}

        return test_data_dict

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        This method returns batch of training data (input and labels).
        Retrieves next batch from dataloader iterator.

        If iterator is empty, it is reset.

        Returns:
            batch: Dictionary containing 'x', input and 'y', label
            of training data batch
        """
        try:
            batch = next(self.training_data_iterators[self._current_teacher_index])
        except StopIteration:
            dataset = self.train_data[self._current_teacher_index]
            self.training_data_iterators[
                self._current_teacher_index
            ] = self._generate_iterator(
                dataset=dataset, batch_size=self._train_batch_size, shuffle=True
            )
            batch = next(self.training_data_iterators[self._current_teacher_index])

        y = batch[1].reshape((self._train_batch_size, -1)).type(torch.FloatTensor)

        batch = {"x": batch[0], "y": y}

        return batch

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        """
        Tells class that task has switched. E.g. this allows get_batch to
        provide data for the relevant task.

        Args:
            new_task: index of new task being trained
        """
        self._current_teacher_index = new_task
