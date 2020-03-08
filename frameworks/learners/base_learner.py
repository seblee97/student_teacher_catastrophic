from models import Teacher, MetaStudent, ContinualStudent, MNISTContinualStudent, MNISTMetaStudent

import copy

import torch

from abc import ABC, abstractmethod

from typing import Dict

class BaseLearner(ABC):

    def __init__(self, config: Dict):

        self.device = config.get("device")
        self.scale_output_backward = config.get(["training", "scale_output_backward"])
        self.learning_rate = config.get(["training", "learning_rate"])
        self.input_dimension = config.get(["model", "input_dimension"])
        self.soft_committee = config.get(["model", "soft_committee"])

        self._setup_student(config=config)

    @abstractmethod
    def _setup_student(self, config):
        raise NotImplementedError("Base class method")

    def forward(self, x):
        return self._student_network(x)

    def forward_all(self, x):
        return self._student_network.forward_all(x)

    def get_trainable_parameters(self):
        if self.scale_output_backward:
            trainable_parameters = [{'params': layer.parameters()} for layer in self._student_network.layers]
            if not self.soft_committee:
                trainable_parameters += [{'params': head.parameters(), 'lr': self.learning_rate / self.input_dimension} for head in self._student_network.heads]
        else:
            trainable_parameters = self._student_network.parameters()   
        return trainable_parameters

    def get_student_network(self):
        return self._student_network

    def set_to_train(self):
        self._student_network = self._student_network.train()
    
    def set_to_eval(self):
        self._student_network = self._student_network.eval()

    @abstractmethod
    def signal_task_boundary_to_learner(self, new_task: int):
        raise NotImplementedError("Base class method")

    # @abstractmethod
    # def compute_generalisation_errors(self, teacher_index=None):
    #     raise NotImplementedError("Base class method")


