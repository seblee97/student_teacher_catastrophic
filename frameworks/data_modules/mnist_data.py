from .base_data_module import BaseData

from typing import Dict

import os

import torch
import torchvision 

from utils import custom_torch_transforms

class MNISTData(BaseData):

    def __init__(self, config: Dict):
        BaseData.__init__(self, config)

        self.train_batch_size = config.get(["training", "train_batch_size"])
        self.pca_input = config.get(["training", "pca_input"])
        self.data_path = config.get("data_path")

        self._load_mnist_data()

    def _load_mnist_data(self):

        file_path = os.path.dirname(os.path.realpath(__file__))
        full_data_path = os.path.join(file_path, self.data_path)

        # transforms to add to data
        transform_list = [
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            custom_torch_transforms.CustomFlatten(),
            custom_torch_transforms.ToFloat()                   
        ]

        base_transform = torchvision.transforms.Compose(transform_list)

        mnist_train_data = torchvision.datasets.MNIST(full_data_path, transform=base_transform, train=True)
        mnist_train_dataloader = torch.utils.data.DataLoader(mnist_train_data, batch_size=60000)

        data_mu = torch.mean(next(iter(mnist_train_dataloader))[0], axis=0)
        data_sigma = torch.std(next(iter(mnist_train_dataloader))[0], axis=0)

        # normalise_transform = torchvision.transforms.Normalize(data_mu, data_sigma)
        # transform_list.insert(0, normalise_transform)
        normalise_transform = custom_torch_transforms.Standardize(data_mu, data_sigma)
        transform_list.append(normalise_transform)

        transform = torchvision.transforms.Compose(transform_list)

        if self.pca_input > 0:
            # note always using train data to generate pca
            mnist_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
            raw_data = mnist_data.train_data.reshape((len(mnist_data), -1))
            pca_output = get_pca(raw_data, num_principal_components=self.pca_input)

            # add application of pca to transforms
            pca_transform = custom_torch_transforms.ApplyPCA(pca_output)

            transform_list.append(pca_transform)
            transform = torchvision.transforms.Compose(transform_list)

        self.mnist_train_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
        self.mnist_test_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=False)

        self.training_dataloader = torch.utils.data.DataLoader(self.mnist_train_data, batch_size=self.train_batch_size, shuffle=True)
        self.training_data_iterator = iter(self.training_dataloader)

        self.test_dataloader = torch.utils.data.DataLoader(self.mnist_test_data, batch_size=self.test_batch_size)

    def get_test_set(self):
        data, labels = next(iter(self.test_dataloader))
        return data, labels

    def get_batch(self):
        try:
            batch_input = next(self.training_data_iterator)[0]
        except:
            self.training_data_iterator = iter(self.training_dataloader)
            batch_input = next(self.training_data_iterator)[0]
        return batch_input