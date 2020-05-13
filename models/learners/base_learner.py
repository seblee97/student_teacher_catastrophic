from abc import ABC, abstractmethod

from typing import Dict, List

from models.networks.base_network import Model

import torch
import torch.nn as nn

import numpy as np

class _BaseLearner(Model):

    def __init__(self, config: Dict):

        Model.__init__(self, config=config, model_type='student')

        self._device = config.get("device")
        self._scale_output_backward = config.get(["training", "scale_output_backward"])
        self._learning_rate = config.get(["training", "learning_rate"])
        self._input_dimension = config.get(["model", "input_dimension"])
        self._soft_committee = config.get(["model", "soft_committee"])

        # self._setup_student(config=config)
        
        if config.get(["task", "loss_type"]) == "classification":
            self.classification_output = True
        elif config.get(["task", "loss_type"]) == 'regression':
            self.classification_output = False
        else:
            raise ValueError("Unknown loss type given in base config")

    @abstractmethod
    def forward_all(self, x) -> List[torch.Tensor]:
        """
        Makes call to student forward, using all heads (used for evaluation)
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def forward_batch_per_task(self, batch_list: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Makes call to student forward, using a different head for each batch in list
        """
        raise NotImplementedError("Base class method")

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """
        To instantiate optimiser, returns relevant (trainable) parameters of student networks
        """
        if self._scale_output_backward:
            trainable_parameters = [{'params': layer.parameters()} for layer in self.layers]
            if not self._soft_committee:
                trainable_parameters += [{'params': head.parameters(), 'lr': self._learning_rate / self._input_dimension} for head in self.heads]
        else:
            trainable_parameters = self.parameters()   
        return trainable_parameters

    def set_to_train(self) -> None:
        """set network to train mode (can affect behaviour of certain torch modules)"""
        self.train()
    
    def set_to_eval(self) -> None:
        """set network to evaluations mode (can affect behaviour of certain torch modules)"""
        self.eval()

    @abstractmethod
    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _construct_output_layers(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def set_task(self, task_index: int):
        raise NotImplementedError("Base class method")
    
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._get_output_from_head(x)

        # threshold 
        if self.classification_output:
            return self._threshold(y)
            
        return y

    def _threshold(self, y):
        return torch.sigmoid(y)
    
    @abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _get_head_weights(self) -> List[torch.Tensor]:
        raise NotImplementedError("Base class method")

