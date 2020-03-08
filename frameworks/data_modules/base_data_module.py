from abc import ABC, abstractmethod

from typing import Dict

class BaseData(ABC):

    def __init__(self, config: Dict):
        
        self.train_batch_size = config.get(["training", "train_batch_size"])
        self.test_batch_size = config.get(["training", "test_batch_size"])
        self.input_dimension = config.get(["model", "input_dimension"])

        self.device = config.get("device")
    
    @abstractmethod
    def get_test_set(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def get_batch(self):
        raise NotImplementedError("Base class method")
                

    # def __init__(self, config: Dict, learner, teachers, data) -> None:
    #     Framework.__init__(self, config)

    #     # generate fixed test data
    #     if self.input_source == 'iid_gaussian':
    #         self.test_input_data = torch.randn(self.test_batch_size, self.input_dimension).to(self.device)
    #     elif self.input_source == 'mnist':
    #         # load mnist test data
    #         self.data_path = config.get("data_path")

    #         test_input_data = iter(load_mnist_data_as_dataloader(
    #             data_path=self.data_path, train=False, batch_size=self.test_batch_size, pca=self.pca_input)
    #             ).next()[0]

    #         self.data_mu = torch.mean(test_input_data, axis=0)
    #         self.data_sigma = torch.std(test_input_data, axis=0)

    #         # edge case of zero sigma (all value equal - no discriminative power)
    #         self.data_sigma[self.data_sigma==0] = 1

    #         # standardise test inputs
    #         self.test_input_data = torch.stack([(d - self.data_mu) / self.data_sigma for d in test_input_data])

    #         # load mnist training data
    #         self.mnist_dataloader = load_mnist_data_as_dataloader(data_path=self.data_path, batch_size=self.train_batch_size, pca=self.pca_input)
    #         # self.mnist_train_x = self.mnist_train_x.type(torch.FloatTensor)
    #         self.training_data_iterator = iter(self.mnist_dataloader)

    #     else:
    #         raise ValueError("Input source type {} not recognised. Please use either iid_gaussian or mnist".format(self.input_source))

    #     self.test_teacher_outputs = [teacher(self.test_input_data) for teacher in self.teachers]