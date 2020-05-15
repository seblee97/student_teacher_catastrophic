import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import torch
import torchvision
from torch.utils.data import DataLoader

from utils import custom_torch_transforms


def visualise_matrix(
    matrix_data: np.ndarray,
    fig_title: str = None,
    normalised: bool = True
) -> matplotlib.figure.Figure:
    """
    Show heatmap of matrix

    :param matrix data: (M x N) numpy array containing matrix data
    :param fig_title: title to be given to figure
    :param normalised: whether or not matrix data is normalised
    :return fig: matplotlib figure with visualisation of matrix data
    """
    fig = plt.figure()
    if normalised:
        plt.imshow(matrix_data, vmin=0, vmax=1)
    else:
        plt.imshow(matrix_data)
    plt.colorbar()
    if fig_title:
        fig.suptitle(fig_title, fontsize=20)
    plt.close()
    return fig


def get_pca(data: np.ndarray, num_principal_components: int):
    # normalise data
    normalised_data = normalize(data.numpy())

    pca = PCA(n_components=num_principal_components)
    pca.fit_transform(normalised_data)
    pca.fit(data)
    return pca


def load_mnist_data_as_dataloader(
    data_path: str,
    batch_size: int,
    train: bool = True,
    pca: int = -1
) -> DataLoader:
    """
    Load mnist image data from specified, convert to grayscaled tensors,
    flatten, return dataloader

    :param data_path: path to data directory
    :param batch_size: batch size for dataloader
    :param train: whether to load train or test data
    :param pca: whether to perform pca (> 0 = number of principle components)
    :return dataloader: pytorch dataloader for mnist training dataset
    """

    file_path = os.path.dirname(os.path.realpath(__file__))
    full_data_path = os.path.join(file_path, data_path)

    # transforms to add to data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        custom_torch_transforms.CustomFlatten(),
        custom_torch_transforms.ToFloat()
    ])

    if pca > 0:
        # note always using train data to generate pca
        mnist_data = torchvision.datasets.MNIST(
            full_data_path, transform=transform, train=True
            )
        raw_data = mnist_data.train_data.reshape((len(mnist_data), -1))
        pca_output = get_pca(raw_data, num_principal_components=pca)

        # add application of pca to transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor(),
            custom_torch_transforms.CustomFlatten(),
            custom_torch_transforms.ApplyPCA(pca_output),
            custom_torch_transforms.ToFloat(),
        ])

    mnist_data = torchvision.datasets.MNIST(
        full_data_path, transform=transform, train=train
        )

    dataloader = torch.utils.data.DataLoader(
        mnist_data, batch_size=batch_size, shuffle=True
        )

    return dataloader


def load_mnist_data(data_path: str, flatten: bool = False, pca: int = -1):
    """
    Load mnist image data from specified, convert to grayscaled tensors.


    :param data_path: path to data directory
    :param flatten: whether to flatten images
    :param pca: whether to perform pca (> 0 = number of principle components)
    :return mnist_train_x: list of training images
    :return mnist_test_y: list of labels for training images
    :return mnist_train_x: list of test images
    :return mnist_test_y: list of labels for test images
    """

    file_path = os.path.dirname(os.path.realpath(__file__))
    full_data_path = os.path.join(file_path, data_path)

    # transforms to add to data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])

    mnist_train = torchvision.datasets.MNIST(
        full_data_path, transform=transform, train=True
        )
    mnist_train_x = mnist_train.train_data
    mnist_train_y = mnist_train.train_labels

    mnist_test = torchvision.datasets.MNIST(
        full_data_path, transform=transform, train=False
        )
    mnist_test_x = mnist_test.test_data
    mnist_test_y = mnist_test.test_labels

    if flatten:
        mnist_train_x = mnist_train_x.view((len(mnist_train_x), -1))
        mnist_test_x = mnist_test_x.view((len(mnist_test_x), -1))

    return mnist_train_x, mnist_train_y, mnist_test_x, mnist_test_y


def tensor_rotate(tensor, degree):
    if degree == 0:
        rotated_tensor = tensor
    elif degree == 90:
        rotated_tensor = torch.from_numpy(
            np.rot90(tensor.detach().numpy(), k=1).copy()
            )
    elif degree == 180:
        rotated_tensor = torch.from_numpy(
            np.rot90(tensor.detach().numpy(), k=1).copy()
            )
    elif degree == 270:
        rotated_tensor = torch.from_numpy(
            np.rot90(tensor.detach().numpy(), k=1).copy()
            )
    elif degree == 360:
        rotated_tensor = torch.from_numpy(
            np.rot90(tensor.detach().numpy(), k=1).copy()
            )
    else:
        raise ValueError("Invalid rotation degree")
    return rotated_tensor
