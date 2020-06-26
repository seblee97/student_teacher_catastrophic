from .base_data_module import _BaseData
from utils import Parameters
from constants import Constants

import torch
import torch.distributions as tdist
from torch.utils.data import DataLoader
import numpy as np

from typing import Dict


class IIDDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_size: int,
        distribution: torch.distributions.Normal,
        input_dimension
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

        self._dataset_size = config.get(["iid_data", "dataset_size"])

        if self._dataset_size != "inf":
            self._dataset = IIDDataset(
                    dataset_size=self._dataset_size,
                    distribution=self._data_distribution,
                    input_dimension=self._input_dimension
                    )
            self._reset_data_iterator()

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
        if self._dataset_size == "inf":
            batch = self._data_distribution.sample(
                        (self._train_batch_size, self._input_dimension)
                    ).to(self._device)
        else:
            try:
                batch = next(iter(self._data_iterator))
            except StopIteration:
                self._reset_data_iterator()
                batch = next(iter(self._data_iterator))
        return {'x': batch}

    def _reset_data_iterator(self):
        self._dataloader = DataLoader(
                self._dataset,
                batch_size=self._train_batch_size,
                shuffle=True
                )
        self._data_iterator = iter(self._dataloader)

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        pass
