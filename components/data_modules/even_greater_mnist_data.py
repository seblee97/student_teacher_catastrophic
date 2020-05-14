from .mnist_data import _MNISTData 
from utils import Parameters

from typing import Dict, List, Union

import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader, TensorDataset

import numpy as np
import copy

class MNISTEvenGreaterData(_MNISTData):

    def __init__(self, config: Parameters):

        self._current_teacher_index: int
        self._num_teachers: int = config.get(["task", "num_teachers"])

        _MNISTData.__init__(self, config)

    def _generate_dataloaders(self) -> None:
        """
        """
        # first dataset 
        even_odd_mapping = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1}
        even_odd_mapping_fn = lambda x: even_odd_mapping[x]
        even_odd_train_dataset = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=True, target_transform=even_odd_mapping_fn)
        even_odd_test_dataset = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=False, target_transform=even_odd_mapping_fn)

        greater_five_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
        greater_five_mapping_fn = lambda x: greater_five_mapping[x]

        greater_five_train_dataset = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=True, target_transform=greater_five_mapping_fn)
        greater_five_test_dataset = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=False, target_transform=greater_five_mapping_fn)
            
        task_train_datasets = [even_odd_train_dataset, greater_five_train_dataset]
        task_test_datasets = [even_odd_test_dataset, greater_five_test_dataset]

        self.task_test_dataloaders = [DataLoader(task_test_data, batch_size=len(task_test_data)) for task_test_data in task_test_datasets]
        self.task_train_dataloaders = [DataLoader(task_train_data, batch_size=self._train_batch_size, shuffle=True) for task_train_data in task_train_datasets]

        self.task_train_iterators = [iter(training_dataloader) for training_dataloader in self.task_train_dataloaders]
        
    def get_test_data(self) -> Union[List[Dict[str, torch.Tensor]], Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        """
        test_set_iterators = [next(iter(test_dataloader)) for test_dataloader in self.task_test_dataloaders]
        full_test_datasets = [test_set_iterator[0] for test_set_iterator in test_set_iterators]
        
        assert torch.equal(full_test_datasets[0], full_test_datasets[1]), "Inputs for test set (and training set) should be identical for both tasks"

        # arbitrarily have one of two dataset inputs as test set inputs
        data = full_test_datasets[0]
        # labels obviously still differ between tasks
        labels = [test_set[1].unsqueeze(1) for test_set in full_test_datasets]

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

        return {'x': batch[0], 'y': batch[1].reshape((self._train_batch_size, -1))}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        self._current_teacher_index = new_task        
