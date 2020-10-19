import abc
import copy
import math
from typing import Callable
from typing import List
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import constants
from utils import custom_activations
from students import base_network


class BaseStudent(base_network.BaseNetwork, abc.ABC):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        loss_type: str,
        nonlinearity: str,
        initialise_outputs: bool,
        soft_committee: bool,
        scale_hidden_lr: bool,
        scale_head_lr: bool,
        num_teachers: int,
        learning_rate: float,
        symmetric_initialisation: bool = False,
        initialisation_std: Optional[float] = None,
    ) -> None:
        self._soft_committee = soft_committee
        self._scale_hidden_lr = scale_hidden_lr
        self._scale_head_lr = scale_head_lr
        self._num_teachers = num_teachers
        self._learning_rate = learning_rate

        super().__init__(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dimension=output_dimension,
            bias=bias,
            loss_type=loss_type,
            nonlinearity=nonlinearity,
            initialise_outputs=initialise_outputs,
            symmetric_initialisation=symmetric_initialisation,
            initialisation_std=initialisation_std,
        )

    def get_trainable_parameters(self):  # TODO: return type
        """To instantiate optimiser, returns relevant (trainable) parameters
        of student network.
        """
        if self._scale_hidden_lr:
            trainable_parameters = [
                {
                    "params": filter(lambda p: p.requires_grad, layer.parameters()),
                    "lr": self._learning_rate / math.sqrt(self._input_dimension),
                }
                for layer in self._layers
            ]
        else:
            trainable_parameters = [
                {"params": filter(lambda p: p.requires_grad, layer.parameters())}
                for layer in self._layers
            ]
        if not self._soft_committee:
            trainable_parameters += self._get_trainable_head_parameters()
        return trainable_parameters

    # @abc.abstractmethod
    # def _construct_output_layers(self):
    #     """Instantiate the output layer."""
    #     pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """This method performs the forward pass. This implements the
        abstract method from the nn.Module base class.

        Args:
            x: input tensor

        Returns:
            y: output of network
        """
        for layer in self._layers:
            x = self._nonlinear_function(self.forward_scaling * layer(x))

        y = self._get_output_from_head(x)

        if self._classification_output:
            return self._threshold(y)

        return y

    def _threshold(self, y: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid threshold."""
        return torch.sigmoid(y)

    def forward_all(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Makes call to student forward, using all heads (used for evaluation)"""
        for layer in self._layers:
            x = self._nonlinear_function(self._forward_scaling * layer(x))
        task_outputs = [head(x) for head in self._heads]
        if self._classification_output:
            return [self._threshold(y) for y in task_outputs]
        return task_outputs

    @abc.abstractmethod
    def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
        """Pass tensor through relevant head of student."""
        pass

    # @abstractmethod
    # def forward_batch_per_task(
    #     self, batch_list: List[Dict[str, torch.Tensor]]
    # ) -> List[torch.Tensor]:
    #     """
    #     Makes call to student forward, using a different head
    #     for each batch in list
    #     """
    #     raise NotImplementedError("Base class method")

    # @abstractmethod
    # def _get_trainable_head_parameters(
    #     self,
    # ) -> List[Dict[str, Iterator[torch.nn.Parameter]]]:
    #     raise NotImplementedError("Base class method")

    # @abstractmethod
    # def signal_task_boundary_to_learner(self, new_task: int) -> None:
    #     raise NotImplementedError("Base class method")

    # @abstractmethod
    # def signal_step_boundary_to_learner(self, step: int, current_task: int):
    #     pass

    # @abstractmethod
    # def set_task(self, task_index: int):
    #     raise NotImplementedError("Base class method")

    # def save_weights(self, path: str):
    #     torch.save(self.state_dict(), path)

    # @abstractmethod
    # def _get_head_weights(self) -> List[torch.Tensor]:
    #     """This method gets the weights of the output layer(s)"""
    #     raise NotImplementedError("Base class method")


# def __init__(self, config: Parameters, model_type: str) -> None:
#     """
#     Initialisation.

#     Args:
#         config: Parameter object wherein network configuration is specified
#         model_type: Describes type of model "teacher_*" or "student" where
#         * should be an integer specifying the index of the teacher.

#     Raises:
#         AssertionError: If model type is not correct format
#     """

#     model_index: Union[int, None]

#     assert (
#         model_type == "student" or "teacher_" in model_type
#     ), "model_type variable has incorrect format. Should be 'student' or 'teacher_i' where i is and integer"

# @abstractmethod
# def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
#     raise NotImplementedError("Base class method")
