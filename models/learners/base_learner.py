from abc import abstractmethod
from typing import Dict
from typing import Iterator
from typing import List

import numpy as np
import torch
import torch.nn as nn

from constants import Constants
from models.networks.base_network import Model
from utils import Parameters


class _BaseLearner(Model):

    def __init__(self, config: Parameters):

        Model.__init__(self, config=config, model_type='student')

        self._device = config.get("device")
        self._scale_head_lr = \
            config.get(["training", "scale_head_lr"])
        self._scale_hidden_lr_backward = config.get(["training", "scale_hidden_lr_backward"])
        self._learning_rate = config.get(["training", "learning_rate"])
        self._input_dimension = config.get(["model", "input_dimension"])
        self._soft_committee = config.get(["model", "soft_committee"])

        self._num_teachers = config.get(["task", "num_teachers"])

        if config.get(["task", "loss_type"]) == "classification":
            self.classification_output = True
        elif config.get(["task", "loss_type"]) == 'regression':
            self.classification_output = False
        else:
            raise ValueError("Unknown loss type given in base config")

        self.heads: nn.ModuleList

    @abstractmethod
    def forward_all(self, x) -> List[torch.Tensor]:
        """
        Makes call to student forward, using all heads (used for evaluation)
        """
        raise NotImplementedError("Base class method")

    @abstractmethod
    def forward_batch_per_task(self,
                               batch_list: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
        """
        Makes call to student forward, using a different head
        for each batch in list
        """
        raise NotImplementedError("Base class method")

    def get_trainable_parameters(self) -> Constants.TRAINABLE_PARAMETERS_TYPE:
        """
        To instantiate optimiser, returns relevant (trainable) parameters
        of student networks
        """
        trainable_parameters: Constants.TRAINABLE_PARAMETERS_TYPE
        # if self._scale_head_lr:
        # hidden layer parameters
        if self._scale_hidden_lr_backward:
            trainable_parameters = [{
                'params': filter(lambda p: p.requires_grad, layer.parameters()),
                'lr': self._learning_rate / np.sqrt(self._input_dimension)
            } for layer in self.layers]
        else:
            trainable_parameters = [{
                'params': filter(lambda p: p.requires_grad, layer.parameters())
            } for layer in self.layers]
        if not self._soft_committee:
            trainable_parameters += self._get_trainable_head_parameters()
        # else:
        #     trainable_parameters = self.parameters()
        return trainable_parameters

    @abstractmethod
    def _get_trainable_head_parameters(self) -> List[Dict[str, Iterator[torch.nn.Parameter]]]:
        raise NotImplementedError("Base class method")

    def set_to_train(self) -> None:
        """set network to train mode
        (can affect behaviour of certain torch modules)
        """
        self.train()

    def set_to_eval(self) -> None:
        """set network to evaluations mode
        (can affect behaviour of certain torch modules)
        """
        self.eval()

    @abstractmethod
    def signal_task_boundary_to_learner(self, new_task: int) -> None:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def signal_step_boundary_to_learner(self, step: int, current_task: int):
        pass

    @abstractmethod
    def _construct_output_layers(self):
        raise NotImplementedError("Base class method")

    @abstractmethod
    def set_task(self, task_index: int):
        raise NotImplementedError("Base class method")

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._get_output_from_head(x)

        if self.classification_output:
            return self._threshold(y)

        return y

    def _threshold(self, y):
        return torch.sigmoid(y)

    def save_weights(self, path: str):
        torch.save(self.state_dict(), path)

    @abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")

    @abstractmethod
    def _get_head_weights(self) -> List[torch.Tensor]:
        """This method gets the weights of the output layer(s)"""
        raise NotImplementedError("Base class method")
