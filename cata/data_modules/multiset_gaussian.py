from typing import Dict
from typing import Union

import torch
import torch.distributions as tdist
import numpy as np
from cata import constants
from cata.data_modules import base_data_module
from torch.utils.data import DataLoader


class IIDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_size: int,
        distribution: torch.distributions.Normal,
        input_dimension,
    ) -> None:
        self._dataset_size = dataset_size
        self._data_distribution = distribution
        self._input_dimension = input_dimension

        self._fixed_dataset = self._data_distribution.sample(
            (self._dataset_size, self._input_dimension)
        )

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx):
        return self._fixed_dataset[idx]


class IIDData(base_data_module.BaseData):
    """Class for generating data drawn i.i.d from unit normal Gaussian."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        input_dimension: int,
        mean: Union[int, float],
        variance: Union[int, float],
        dataset_size: Union[str, int],
        mask_proportion: Union[int, float]
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            input_dimension=input_dimension,
        )

        self._data_distribution = tdist.Normal(mean, variance)

        self._mask_dimension = int(input_dimension*mask_proportion)
        self._dataset_size = dataset_size

        if self._dataset_size != constants.INF:
            self._dataset = IIDDataset(
                dataset_size=self._dataset_size,
                distribution=self._data_distribution,
                input_dimension=self._input_dimension,
            )
            self._reset_data_iterator()

    def get_test_data(self, teacher) -> Dict[str, torch.Tensor]:
        """Give fixed test data set (input data only)."""
        if teacher > 2:
            raise ValueError(
                "Multiset Gaussian distribution currently only supported for 2 teachers"
            )
        if teacher == 0:
            test_input_data = self._data_distribution.sample(
                (self._test_batch_size, self._input_dimension)
            )
            return test_input_data

        if teacher == 1:
            test_input_data = self._get_masked_batch(self._test_batch_size, self._input_dimension, 1)
            return test_input_data

    def get_batch(self, teacher, replay) -> Dict[str, torch.Tensor]:
        """Returns batch of training data (input only)"""
        # For training, we return unmasked data for teacher 0, masked data for teacher 1, and inverse masked data for replay
        batch = None
        if teacher == 0 and not replay:
            if self._dataset_size == constants.INF:
                batch = self._get_infinite_dataset_batch()
            else:
                batch = self._get_finite_dataset_batch()
        if teacher == 1:
            if self._dataset_size == constants.INF:
                batch = self._get_infinite_dataset_batch(1)
            else:
                batch = self._get_finite_dataset_batch()
        if teacher == 0 and replay:
            if self._dataset_size == constants.INF:
                batch = self._get_infinite_dataset_batch(2)
            else:
                batch = self._get_finite_dataset_batch()
        return {constants.X: batch}

    def _get_finite_dataset_batch(self) -> torch.Tensor:
        try:
            batch = next(iter(self._data_iterator))
        except StopIteration:
            self._reset_data_iterator()
            batch = next(iter(self._data_iterator))
        return batch

    def _get_infinite_dataset_batch(self, masking=0) -> torch.Tensor:
        if masking == 0:
            batch = self._data_distribution.sample(
                (self._train_batch_size, self._input_dimension)
            )
            return batch

        return self._get_masked_batch(self._train_batch_size, self._input_dimension, masking)
        # sample = [self._mask(self._data_distribution.sample((1, self._input_dimension)), masking) for i in range(self._train_batch_size)]
        # batch = torch.cat(sample)
        # return batch

    def _reset_data_iterator(self):
        self._dataloader = DataLoader(
            self._dataset, batch_size=self._train_batch_size, shuffle=True
        )
        self._data_iterator = iter(self._dataloader)

    def _get_masked_batch(self, batch_size, dimension, masking):
        sample = [self._mask(self._data_distribution.sample((1, dimension)), masking) for i in range(batch_size)]

        batch = torch.stack(sample)
        return batch

    def _mask(self, vector, masking):
        if masking == 0:
            return vector

        # Simple masking for test data
        if masking == 1:
            split = vector.split(self._mask_dimension, dim=1)
            negative = split[0].apply_(lambda x: (-1*x) if x > 0 else x)
            return torch.cat([negative[0], split[1][0]])

        # Masking for replay
        if masking == 2:
            # Currently, let's replay only positive vectors (overlap = 0) MUST BE CHANGED.
            split = vector.split(self._mask_dimension, dim=1)
            positive = split[0].apply_(lambda x: (-1*x) if x < 0 else x)
            return torch.cat([positive[0], split[1][0]])
            """
            for i in range(len(vector[0][self._half_dimension:self._input_dimension - 1])):
                if vector[0][i] > 0:
                    vector[0][i] *= -1
            return vector
            """

            """
            Old code (randomness)
            tries = 0
            second_half = vector[0][self._half_dimension:self._input_dimension - 1]
            less_than_zero = sum([1 if x <= 0 else 0 for x in second_half])
            attempts = [[less_than_zero, vector]]

            while tries < 50 and less_than_zero < self._mask_number:
                vector = self._data_distribution.sample((1, self._input_dimension))
                second_half = vector[0][self._half_dimension:self._input_dimension-1]
                less_than_zero = sum([1 if x <= 0 else 0 for x in second_half])
                tries += 1
                attempts.append([less_than_zero, vector])
            mx = max(attempts, key=lambda x: x[0])
            return mx[1]
            """


