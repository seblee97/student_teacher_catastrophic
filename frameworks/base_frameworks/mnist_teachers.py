from models import Teacher

from frameworks import MNIST

import torch
import torchvision

import random
import os

from typing import List

class MNISTTeachers(MNIST):

    def __init__(self, config):
        MNIST.__init__(self, config)

    def _setup_teachers(self, config):
        """Instantiate all teachers - in this case MNIST binary classification categories"""
        self.data_path = os.path.join(os.getcwd(), config.get('data_path'))
        train_batch_size = config.get(['training', 'train_batch_size'])
        test_batch_size = config.get(['training', 'test_batch_size'])

        self.num_teachers = config.get(['task', 'num_teachers'])
        self.mnist_teacher_classes = config.get(['training', 'teachers'])

        if self.num_teachers != len(self.mnist_teacher_classes):
            raise AssertionError("Number of teachers specified in base config is not compatible with\
                number of MNIST classifiers specified.")

        # transforms to add to data
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])

        mnist_train = torchvision.datasets.MNIST(self.data_path, transform=transform, train=True)
        self.mnist_train_x = mnist_train.train_data
        self.mnist_train_y = mnist_train.train_labels

        mnist_test = torchvision.datasets.MNIST(self.data_path, transform=transform, train=False)
        mnist_test_x = mnist_test.test_data
        mnist_test_y = mnist_test.test_labels

        # initialise different binary classifiers
        self.teachers = [_ for _ in range(self.num_teachers)]

        self.test_data_x = []
        self.test_data_y = []

        for t, task in enumerate(self.mnist_teacher_classes):
            self._reset_batch(t, task)

        # test data sets
        for task in self.mnist_teacher_classes:

            d1_test = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([0]).type(torch.LongTensor)) for (i, x) in enumerate(mnist_test_x) if mnist_test_y[i] == task[0]]
            d2_test = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([1]).type(torch.LongTensor)) for (i, x) in enumerate(mnist_test_x) if mnist_test_y[i] == task[1]]
            task_test_data = d1_test + d2_test
            random.shuffle(task_test_data)
            
            task_test_inputs = torch.stack([d[0] for d in task_test_data]).to(self.device)
            task_test_labels = torch.stack([d[1] for d in task_test_data]).squeeze().to(self.device)
            self.test_data_x.append(task_test_inputs)
            self.test_data_y.append(task_test_labels)

    def _reset_batch(self, task_index: int, classes: List[int]):
        d1_train = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([0]).type(torch.LongTensor)) for (i, x) in enumerate(self.mnist_train_x) if self.mnist_train_y[i] == classes[0]]
        d2_train = [(torch.flatten(x).type(torch.FloatTensor), torch.Tensor([1]).type(torch.LongTensor)) for (i, x) in enumerate(self.mnist_train_x) if self.mnist_train_y[i] == classes[1]]
        task_train_data = d1_train + d2_train
        random.shuffle(task_train_data)
        self.teachers[task_index] = task_train_data

    def _signal_task_boundary_to_teacher(self, new_task: int):
        pass

    def _signal_step_boundary_to_teacher(self, step: int, current_task: int):
        pass
    