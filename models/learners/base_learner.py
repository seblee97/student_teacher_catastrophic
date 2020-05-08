from abc import ABC, abstractmethod

from typing import Dict, List

import torch
import torch.nn as nn

class _BaseLearner(ABC):

    def __init__(self, config: Dict):

        self._device = config.get("device")
        self._scale_output_backward = config.get(["training", "scale_output_backward"])
        self._learning_rate = config.get(["training", "learning_rate"])
        self._input_dimension = config.get(["model", "input_dimension"])
        self._soft_committee = config.get(["model", "soft_committee"])

        self._setup_student(config=config)

    @abstractmethod
    def _setup_student(self, config: Dict) -> None:
        """Initialise student networks"""
        raise NotImplementedError("Base class method")

    def forward(self, x) -> torch.Tensor:
        """
        Call forward method of student network 
        
        (Note: this class is not an instance of torch.nn.Module 
        so this forward is not part of that API)
        """
        return self._student_network(x)

    def forward_all(self, x) -> List[torch.Tensor]:
        """
        Makes call to student forward, using all heads (used for evaluation)
        """
        return self._student_network.forward_all(x)

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        To instantiate optimiser, returns relevant (trainable) parameters of student networks
        """
        if self._scale_output_backward:
            trainable_parameters = [{'params': layer.parameters()} for layer in self._student_network.layers]
            if not self._soft_committee:
                trainable_parameters += [{'params': head.parameters(), 'lr': self._learning_rate / self._input_dimension} for head in self._student_network.heads]
        else:
            trainable_parameters = self._student_network.parameters()   
        return trainable_parameters

    def get_student_network(self) -> nn.Module:
        """getter for student network object"""
        return self._student_network

    def set_to_train(self) -> None:
        """set network to train mode (can affect behaviour of certain torch modules)"""
        self._student_network = self._student_network.train()
    
    def set_to_eval(self) -> None:
        """set network to evaluations mode (can affect behaviour of certain torch modules)"""
        self._student_network = self._student_network.eval()

    @abstractmethod
    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        raise NotImplementedError("Base class method")
