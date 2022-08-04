from typing import Dict
from typing import Union

import torch
import scipy.stats
import numpy as np
from cata import constants
from cata.data_modules import base_data_module
from torch.utils.data import DataLoader

class ExponnormDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_size: int,
        distribution: scipy.stats.exponnorm(0.5),
        input_dimension,
    ) -> None:
        self._dataset_size = dataset_size
        self._data_distribution = distribution
        self._input_dimension = input_dimension

        self._fixed_dataset = [self._data_distribution.rvs(size=self._input_dimension) for i in range(self._input_dimension)]
        self._fixed_dataset = torch.FloatTensor(np.array(self._fixed_dataset))

    def __len__(self):
        return self._dataset_size

    def __getitem__(self, idx):
        return self._fixed_dataset[idx]


class ExponnormData(base_data_module.BaseData):
    """Class for generating data convolved from an exponential and normal distribution."""

    def __init__(
        self,
        train_batch_size: int,
        test_batch_size: int,
        input_dimension: int,
        mean: Union[int, float],
        variance: Union[int, float],
        k: Union[int, float],
        kRange: Union[int, float],
        meanRange: Union[int, float],
        stdRange: Union[int, float],
        dataset_size: Union[str, int],
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            input_dimension=input_dimension,
        )

        self._data_distribution = scipy.stats.exponnorm(k, loc=mean, scale=np.sqrt(variance))

        self._dataset_size = dataset_size

        if self._dataset_size != constants.INF:
            self._dataset = ExponnormDataset(
                dataset_size=self._dataset_size,
                distribution=self._data_distribution,
                input_dimension=self._input_dimension,
            )
            self._reset_data_iterator()

    def get_random_exponnorm(self, kran, mran, stdran):
        """generates a randomised exponnorm distribution"""
        mean = np.random.default_rng().uniform(-1*mran, mran)
        k = -1*np.random.default_rng().uniform(-1*kran, 0) #picks non-zero random value in range
        std = -1*np.random.default_rng().uniform(-1*stdran,0) #picks non-zero random value in range
        self._data_distribution = scipy.stats.exponnorm(k, loc=mean, scale=std)
        #consider renaming the distribution
        
    def sample(self, num, dim) -> torch.Tensor:
        """Generates a num x dim array of samples from the distribution"""

        self.get_random_exponnorm(self.kRange, self.meanRange, self.stdRange) #each batch will have a different gaussian
        sample = [self._data_distribution.rvs(size=dim) for i in range(num)]
        sample = torch.FloatTensor(np.array(sample))

        return sample

    def get_test_data(self) -> Dict[str, torch.Tensor]:
        """Give fixed test data set (input data only)."""

        test_input_data = self.sample(self._test_batch_size, self._input_dimension)
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
        batch = self.sample(self._train_batch_size, self._input_dimension)

        return batch

    def _reset_data_iterator(self):
        self._dataloader = DataLoader(
            self._dataset, batch_size=self._train_batch_size, shuffle=True
        )
        self._data_iterator = iter(self._dataloader)
