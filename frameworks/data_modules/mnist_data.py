from .base_data_module import _BaseData

from typing import Dict

import os

import torch
import torchvision 

from utils import custom_torch_transforms, get_pca

class MNISTData(_BaseData):

    """Class for dealing with data generated from MNIST images"""

    def __init__(self, config: Dict):
        _BaseData.__init__(self, config)

        self._pca_input = config.get(["training", "pca_input"])
        self._data_path = config.get("data_path")

        self._load_mnist_data()

    def _load_mnist_data(self) -> None:

        file_path = os.path.dirname(os.path.realpath(__file__))
        full_data_path = os.path.join(file_path, self._data_path)

        # transforms to add to data
        transform_list = [
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            custom_torch_transforms.CustomFlatten(),
            custom_torch_transforms.ToFloat()                   
        ]

        base_transform = torchvision.transforms.Compose(transform_list)

        # load dataset to compute channel statistics
        mnist_train_data = torchvision.datasets.MNIST(full_data_path, transform=base_transform, train=True)
        mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_data, batch_size=60000)

        # evaluate channel mean and std
        data_mu = torch.mean(next(iter(mnist_train_dataloader))[0], axis=0)
        data_sigma = torch.std(next(iter(mnist_train_dataloader))[0], axis=0)

        # add normalisation to transforms
        normalise_transform = custom_torch_transforms.Standardize(data_mu, data_sigma)
        transform_list.append(normalise_transform)

        transform = torchvision.transforms.Compose(transform_list)

        if self._pca_input > 0:
            # note always using train data to generate pca
            mnist_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
            raw_data = mnist_data.train_data.reshape((len(mnist_data), -1))
            pca_output = get_pca(raw_data, num_principal_components=self._pca_input)

            # add application of pca to transforms
            pca_transform = custom_torch_transforms.ApplyPCA(pca_output)

            transform_list.append(pca_transform)
            transform = torchvision.transforms.Compose(transform_list)

        self.mnist_train_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
        self.mnist_test_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=False)

        self.training_dataloader = torch.utils.data.DataLoader(self.mnist_train_data, batch_size=self._train_batch_size, shuffle=True)
        self.training_data_iterator = iter(self.training_dataloader)

        self.test_dataloader = torch.utils.data.DataLoader(self.mnist_test_data, batch_size=self._test_batch_size)

    def get_test_set(self) -> (torch.Tensor, torch.Tensor):
        """
        returns fixed test data set (data and labels)
        
        return data: test data inputs
        return labels: corresponding labels for test data inputs
        """
        data, labels = next(iter(self.test_dataloader))
        return data, labels

    def get_batch(self) -> torch.Tensor:
        """
        returns batch of training data. Retrieves next batch from dataloader iterator.
        If iterator is empty, it is reset.

        return batch_input:  
        """
        try:
            batch_input = next(self.training_data_iterator)[0]
        except:
            self.training_data_iterator = iter(self.training_dataloader)
            batch_input = next(self.training_data_iterator)[0]
        return batch_input