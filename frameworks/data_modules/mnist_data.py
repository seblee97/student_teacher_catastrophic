from .base_data_module import BaseData

from typing import Dict

class MNISTData(BaseData):

    def __init__(self, config: Dict):
        super(BaseData).__init__(self)

    def get_test_data(self):
        pass

    def get_batch(self):
        try:
            batch_input = next(self.training_data_iterator)[0].to(self.device)
        except:
            self.training_data_iterator = iter(self.mnist_dataloader)
            batch_input = next(self.training_data_iterator)[0].to(self.device)

        # normalise input
        batch_input = (batch_input - self.data_mu) / self.data_sigma