from typing import Dict
from typing import Union

import torch
import torch.distributions as tdist
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
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            input_dimension=input_dimension,
        )

        self._data_distribution = tdist.Normal(mean, variance)

        self._dataset_size = dataset_size

        if self._dataset_size != constants.INF:
            self._dataset = IIDDataset(
                dataset_size=self._dataset_size,
                distribution=self._data_distribution,
                input_dimension=self._input_dimension,
            )
            self._reset_data_iterator()

    def get_replay_data(self):
        """Pick samples for replay from training dataset"""
    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """Give fixed test data set (input data only)."""
        test_input_data = self._data_distribution.sample(
            (self._test_batch_size, self._input_dimension)
        )

        test_data_dict = {constants.X: test_input_data}

        return test_data_dict

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Returns batch of training data (input only)"""
        if self._dataset_size == constants.INF:
            batch = self._get_infinite_dataset_batch()
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

    def _get_infinite_dataset_batch(self) -> torch.Tensor:
        batch = self._data_distribution.sample(
            (self._train_batch_size, self._input_dimension)
        )
        return batch

    def _reset_data_iterator(self):
        self._dataloader = DataLoader(
            self._dataset, batch_size=self._train_batch_size, shuffle=True
        )
        self._data_iterator = iter(self._dataloader)
