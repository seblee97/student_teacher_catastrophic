from .mnist_data import _MNISTData 

from typing import Dict, List

import torch
import torchvision
from torch.utils.data import Subset, ConcatDataset, DataLoader, TensorDataset

import numpy as np
import copy

class MNISTEvenGreaterData(_MNISTData):

    def __init__(self, config: Dict):

        self._current_teacher_index = None
        self._num_teachers = config.get(["task", "num_teachers"])

        _MNISTData.__init__(self, config)

    def map_dataset_targets(self, original_dataset, mapping: Dict):

        mapped_dataset = copy.deepcopy(original_dataset)
        original_targets = original_dataset.targets
        new_targets = copy.deepcopy(original_targets) 

        for digit, new_label in mapping.items():
            new_targets[original_targets is digit] = new_label

        mapped_dataset.targets = new_targets

        return mapped_dataset

        # return TensorDataset(original_features, new_targets)

    def _generate_dataloaders(self) -> None:

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
        
    def get_test_data(self) -> (torch.Tensor, List[torch.Tensor]):
        test_sets = [next(iter(test_dataloader)) for test_dataloader in self.task_test_dataloaders]
        
        assert torch.equal(test_sets[0][0], test_sets[1][0]), "Inputs for test set (and training set) should be identical for both tasks"

        # arbitrarily have one of two dataset inputs as test set inputs
        data = test_sets[0][0]
        # labels obviously still differ between tasks
        labels = [test_set[1].unsqueeze(1) for test_set in test_sets]

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
