from .mnist_data import _MNISTData 

import torch
import torchvision

from typing import Dict, List

class MNISTStreamData(_MNISTData):

    def __init__(self, config: Dict):
        _MNISTData.__init__(self, config)

    def _generate_dataloaders(self) -> None:

        self.train_data = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=True)
        self.test_data = torchvision.datasets.MNIST(self._full_data_path, transform=self.transform, train=False)

        self.training_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self._train_batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_data, batch_size=len(self.test_data))

        self.training_data_iterator = iter(self.training_dataloader)

    def get_test_set(self) -> (torch.Tensor, List[torch.Tensor]):
        data, labels = next(iter(self.test_dataloader))
        return {'x': data}

    def get_batch(self) -> Dict[str, torch.Tensor]:
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

        return {'x': batch_input}

    def signal_task_boundary_to_data_generator(self, new_task: int) -> None:
        pass