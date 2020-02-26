import numpy as np
import os
#import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import torch 
import torchvision

from typing import List, Tuple

from utils import custom_torch_transforms

def visualise_matrix(matrix_data: np.ndarray, fig_title: str=None, normalised: bool=True):
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
    reduced_data = pca.fit_transform(normalised_data)
    pca.fit(data)
    # signular_values = pca.singular_values_
    # components = pca.components_
    # return reduced_data, signular_values, components
    return pca

def load_mnist_data_as_dataloader(data_path: str, batch_size: int, train:bool=True, pca: int=-1):
    """
    Load mnist image data from specified, convert to grayscaled tensors, flatten, return dataloader

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
        mnist_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
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

    mnist_data = torchvision.datasets.MNIST(full_data_path, transform=transform, train=train)

    dataloader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    return dataloader

def load_mnist_data(data_path: str, flatten: bool=False, pca: int=-1):
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

    mnist_train = torchvision.datasets.MNIST(full_data_path, transform=transform, train=True)
    mnist_train_x = mnist_train.train_data
    mnist_train_y = mnist_train.train_labels

    mnist_test = torchvision.datasets.MNIST(full_data_path, transform=transform, train=False)
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
        rotated_tensor = torch.from_numpy(np.rot90(tensor.detach().numpy(), k=1).copy())
    elif degree == 180:
        rotated_tensor = torch.from_numpy(np.rot90(tensor.detach().numpy(), k=1).copy())
    elif degree == 270:
        rotated_tensor = torch.from_numpy(np.rot90(tensor.detach().numpy(), k=1).copy())
    elif degree == 360:
        rotated_tensor = torch.from_numpy(np.rot90(tensor.detach().numpy(), k=1).copy())
    else:
        raise ValueError("Invalid rotation degree")
    return rotated_tensor

def get_binary_classification_datasets(x_data: List, y_data: List, task_classes: List[int], rotations: List[int]=None) -> List[Tuple]:
    """
    Filter dataset for specific labels

    :param x_data: mnist image input
    :param y_data: corresponding labels
    :param task_classes: list of label indices defining the task
    :param rotations: list of angles by which to rotate images
    :return task_data: filtered dataset
    """ 
    if rotations:
        d1_train = [(torch.flatten(tensor_rotate(x, rotations[0])).type(torch.FloatTensor), torch.Tensor([0]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task_classes[0]]
        d2_train = [(torch.flatten(tensor_rotate(x, rotations[1])).type(torch.FloatTensor), torch.Tensor([1]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task_classes[1]]
    else:
        d1_train = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([0]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task_classes[0]]
        d2_train = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([1]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task_classes[1]]
    task_data = d1_train + d2_train
    random.shuffle(task_data)
    return task_data

# def get_binary_classification_datasets(x_data: List, y_data: List, task_classes: List[List[int]]):

#     test_data_x = []
#     test_data_y = []

#     # test data sets
#     for task in task_classes:

#         d1_test = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([0]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task[0]]
#         d2_test = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([1]).type(torch.LongTensor)) for (i, x) in enumerate(x_data) if y_data[i] == task[1]]
#         task_test_data = d1_test + d2_test
#         random.shuffle(task_test_data)
        
#         task_test_inputs = torch.stack([d[0] for d in task_test_data])
#         task_test_labels = torch.stack([d[1] for d in task_test_data]).squeeze()
#         test_data_x.append(task_test_inputs)
#         test_data_y.append(task_test_labels)

#     return test_data_x, test_data_y