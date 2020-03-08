import torch
import torchvision

import random
import os
import copy

from typing import List

from models import Teacher, MetaStudent, ContinualStudent
from utils import load_mnist_data, get_binary_classification_datasets

from .base_teacher import _BaseTeacher

class MNISTTeachers(_BaseTeacher):

    def __init__(self, config):
        _BaseTeacher.__init__(self, config)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    qqqqqqq
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

        # load mnist data
        self.mnist_train_x, self.mnist_train_y, mnist_test_x,  mnist_test_y = load_mnist_data(data_path=self.data_path)

        # get fixed test dataset (filtered for tasks specified)
        test_data = [get_binary_classification_datasets(
            x_data=mnist_test_x, y_data=mnist_test_y, task_classes=task_classes, rotations=rotations
            ) for task_classes, rotations in zip(self.mnist_teacher_classes, self.rotations)]
        self.test_data_x, self.test_data_y = [], []
        for task_test_data in test_data: 
            task_test_inputs = torch.stack([d[0] for d in task_test_data]).to(self.device)
            task_test_labels = torch.stack([d[1] for d in task_test_data]).squeeze().to(self.device)
            self.test_data_x.append(task_test_inputs)
            self.test_data_y.append(task_test_labels)

        # initialise different binary classifiers
        self.teachers = [_ for _ in range(self.num_teachers)]

        # filter data for relevant images for each task
        for t, task in enumerate(self.mnist_teacher_classes):
            teacher_data = get_binary_classification_datasets(self.mnist_train_x, self.mnist_train_y, task, rotations=self.rotations[t])
            self.teachers[t] = teacher_data

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
    