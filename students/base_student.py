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


class BaseStudent(abc.ABC, nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: List[int],
        output_dimension: int,
        bias: bool,
        soft_committee: bool,
        num_teachers: int,
        loss_type: str,
        learning_rate: float,
        scale_head_lr: bool,
        scale_hidden_lr: bool,
        nonlinearity: str,
        symmetric_initialisation: bool,
        initialise_student_outputs: bool,
        initialisation_std: Optional[float] = None,
    ):
        self._input_dimension = input_dimension
        self._hidden_dimensions = hidden_dimensions
        self._output_dimension = output_dimension
        self._bias = bias
        self._soft_committee = soft_committee
        self._num_teachers = num_teachers
        self._classification_output = loss_type == constants.Constants.CLASSIFICATION
        self._learning_rate = learning_rate
        self._scale_head_lr = scale_head_lr
        self._scale_hidden_lr = scale_hidden_lr
        self._nonlinearity = nonlinearity
        self._initialise_student_outputs = initialise_student_outputs
        self._initialisation_std = initialisation_std
        self._symmetric_initialisation = symmetric_initialisation

        super().__init__()

        self._nonlinear_function = self._get_nonlinear_function()
        self._construct_layers()

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

    def _get_nonlinear_function(self) -> Callable:
        """Makes the nonlinearity function specified by the config.

        Returns:
            nonlinear_function: Callable object, the nonlinearity function

        Raises:
            ValueError: If nonlinearity provided in config is not recognised
        """
        if self._nonlinearity == constants.Constants.RELU:
            nonlinear_function = F.relu
        elif self._nonlinearity == constants.Constants.SIGMOID:
            nonlinear_function = torch.sigmoid
        elif self._nonlinearity == constants.Constants.SCALED_ERF:
            nonlinear_function = custom_activations.scaled_erf_activation
        elif self._nonlinearity == constants.Constants.LINEAR:
            nonlinear_function = custom_activations.linear_activation
        else:
            raise ValueError(f"Unknown non-linearity: {self._nonlinearity}")
        return nonlinear_function

    def _construct_layers(self) -> None:
        """Instantiate layers (input, hidden and output) according to
        dimensions specified in configuration. Note this method makes a call to
        the abstract _construct_output_layers, which is implemented by the
        child.
        """
        self._layers = nn.ModuleList([])

        layer_dimensions = [self._input_dimension] + self._hidden_dimensions

        for layer_size, next_layer_size in zip(layer_dimensions, layer_dimensions[1:]):
            layer = nn.Linear(layer_size, next_layer_size, bias=self._bias)
            self._initialise_weights(layer)
            self._layers.append(layer)

        self._construct_output_layers()

    def _initialise_weights(self, layer: nn.Module) -> None:
        """In-place weight initialisation for a given layer in accordance with configuration.

        Args:
            layer: the layer to be initialised.
        """
        if self._initialisation_std is not None:
            nn.init.normal_(layer.weight, std=self._initialisation_std)
            if self._bias:
                nn.init.normal_(layer.bias, std=self._initialisation_std)
        if self._symmetric_initialisation:
            # copy first component of weight matrix to others to ensure zero overlap
            base_parameters = copy.deepcopy(
                layer.state_dict()[constants.Constants.WEIGHT][0]
            )
            for dim in range(1, len(layer.state_dict()[constants.Constants.WEIGHT])):
                layer.state_dict()[constants.Constants.WEIGHT][dim] = base_parameters

    @abc.abstractmethod
    def _construct_output_layers(self):
        """Instantiate the output layer."""
        pass

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
    # def _get_output_from_head(self, x: torch.Tensor) -> torch.Tensor:
    #     raise NotImplementedError("Base class method")

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
