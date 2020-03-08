from .base_data_module import BaseData

import torch

class IIDData(BaseData):

    def __init__(self, config):
        BaseData.__init__(self, config)

    def get_test_set(self):
        test_input_data = torch.randn(self.test_batch_size, self.input_dimension).to(self.device)
        return test_input_data, None 

    def get_batch(self):
        batch = torch.randn(self.train_batch_size, self.input_dimension).to(self.device)
        return batch