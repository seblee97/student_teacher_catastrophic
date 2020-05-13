from .mnist_data import _MNISTData 

from typing import Dict, List

import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader

import numpy as np
import copy

class PureMNISTData(_MNISTData):

    def __init__(self, config: Dict, override_batch_size: int=None):

        self.override_batch_size = override_batch_size

        self._mnist_teacher_classes = config.get(['pure_mnist', 'teacher_digits'])
        self._current_teacher_index = None

        _MNISTData.__init__(self, config)

        if self._num_teachers != len(self._mnist_teacher_classes):
            raise AssertionError("Number of teachers specified in base config is not compatible with\
                number of MNIST classifiers specified.")

    def filter_dataset_by_target(self, unflitered_data, target_to_keep: int, train: bool, new_label: int=None):
        """generate subset of dataset filtered by target"""
        targets = unflitered_data.targets

        indices_to_keep = targets == target_to_keep
        filtered_dataset_indices = np.arange(len(targets))[indices_to_keep]

        if train:
            train_dataset_copy = copy.deepcopy(unflitered_data)
            for ind in filtered_dataset_indices:
                train_dataset_copy.targets[ind] = torch.Tensor([new_label])
            return Subset(train_dataset_copy, filtered_dataset_indices)
        else:  
            return Subset(unflitered_data, filtered_dataset_indices)

    def _generate_dataloaders(self) -> None:

        full_train_data = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=True)
        full_test_data = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=False)

        task_train_datasets = []
        task_test_datasets = []

        for t, task_classes in enumerate(self._mnist_teacher_classes):
            
            # get filtered training sets
            train_target_filtered_datasets = [self.filter_dataset_by_target(
                unflitered_data=full_train_data, target_to_keep=target, train=True, new_label=st
                ) for st, target in enumerate(task_classes)]
            train_task_dataset = ConcatDataset(train_target_filtered_datasets)

            # get filtered test sets
            test_target_filtered_datasets = [
                self.filter_dataset_by_target(unflitered_data=full_test_data, target_to_keep=target, train=False
                ) for target in task_classes]
            test_task_dataset = ConcatDataset(test_target_filtered_datasets)

            task_train_datasets.append(train_task_dataset)
            task_test_datasets.append(test_task_dataset)

        self.task_test_dataloaders = [DataLoader(task_test_data, batch_size=len(task_test_data)) for task_test_data in task_test_datasets]
        
        if self.override_batch_size is not None:
            bs = self.override_batch_size
        else:
            bs = self._train_batch_size
        self.task_train_dataloaders = [DataLoader(task_train_data, batch_size=bs, shuffle=True) for task_train_data in task_train_datasets]
        self.task_train_iterators = [iter(training_dataloader) for training_dataloader in self.task_train_dataloaders]
        
        # train_dataset = Subset(dataset, train_idx)

        # self.training_dataloader = torch.utils.data.DataLoader(self.mnist_train_data, batch_size=self._train_batch_size, shuffle=True)
        # self.training_data_iterator = iter(self.training_dataloader)

    def get_test_data(self) -> (torch.Tensor, List[torch.Tensor]):
        test_sets = [next(iter(test_dataloader)) for test_dataloader in self.task_test_dataloaders]
        data = torch.cat([test_set[0] for test_set in test_sets])
        labels = [test_set[1] for test_set in test_sets]
        return {'x': data, 'y': labels}
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        returns batch of training data. Retrieves next batch from dataloader iterator.
        If iterator is empty, it is reset. Returns both input and label.

        return batch_input:  
        """
        try:
            batch = next(self.task_train_iterators[self._current_teacher_index])
        except:
            self.task_train_iterators[self._current_teacher_index] = iter(self.task_train_dataloaders[self._current_teacher_index])
            batch = next(self.task_train_iterators[self._current_teacher_index])

        return {'x': batch[0], 'y': batch[1]}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        self._current_teacher_index = new_task        
