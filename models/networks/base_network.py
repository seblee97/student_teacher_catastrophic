import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy
import numpy as np

from typing import Union, Callable

from abc import ABC, abstractmethod

from utils import Parameters, linear_function


class Model(nn.Module, ABC):
    """
    Multi-layer non-linear neural network class. For use in
    student-teacher framework. This is an abstract base class.

    Methods that need to be implemented by child class:

    _construct_output_layers
    _output_forward

    This base class does the relevant initialisation and makes the
    forward up to the last layer, which is implemented by the child.
    """
    def __init__(self, config: Parameters, model_type: str) -> None:
        """
        Initialisation.

        Args:
            config: Parameter object wherein network configuration is specified
            model_type: Describes type of model "teacher_*" or "student" where
            * should be an integer specifying the index of the teacher.

        Raises:
            AssertionError: If model type is not correct format
        """

        model_index: Union[int, None]

        assert (model_type == 'student' or 'teacher_' in model_type), \
            (
                "model_type variable has incorrect format. Should be 'student'"
                " or 'teacher_i' where i is and integer"
            )

        if 'teacher' in model_type:
            self.model_type, model_index_str = model_type.split("_")
            self.model_index = int(model_index_str)
        else:
            self.model_type = model_type

        self._extract_parameters(config=config)

        self.nonlinear_function = self._get_nonlinear_function(config=config)

        super(Model, self).__init__()

        self._construct_layers()

    def _extract_parameters(self, config: Parameters):
        """
        This method extracts the relevant parameters from the config
        and makes them parameters of this class for easier access.

        Args:
            config: Overall Parameter object

        Raises:
            ValueError: If field does not exist in config object
        """
        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        self.hidden_dimensions = \
            config.get(["model", "{}_hidden_layers".format(self.model_type)])
        self.initialisation_std = \
            config.get(
                ["model", "{}_initialisation_std".format(self.model_type)]
                )
        self.symmetric_student_initialisation = \
            config.get(
                ["model", "symmetric_student_initialisation"]
            )
        self.bias = \
            config.get(["model", "{}_bias_parameters".format(self.model_type)])
        self.soft_committee = config.get(["model", "soft_committee"])
        self.learner_configuration = \
            config.get(["task", "learner_configuration"])
        self.num_teachers = config.get(["task", "num_teachers"])
        self.label_task_boundaries = \
            config.get(["task", "label_task_boundaries"])
        self.initialise_student_outputs = \
            config.get(["model", "initialise_student_outputs"])
        self.forward_scaling = 1 / math.sqrt(self.input_dimension)
        self.normalise_teachers = config.get(["model", "normalise_teachers"])

    def _get_nonlinear_function(self, config: Parameters) -> Callable:
        """
        This method makes the nonlinearity function specified by the config.
        Note for teachers, a list of nonlinearities is provided - one for each
        teacher.

        Args:
            config: Overall Parameter object

        Returns:
            nonlinear_function: Callable object, the nonlinearity function

        Raises:
            ValueError: If nonlinearity provided in config is not recognised
        """

        if self.model_type == 'teacher':
            all_teacher_nonlinearities = \
                config.get(["model", "teacher_nonlinearities"])
            self.nonlinearity_name = \
                all_teacher_nonlinearities[self.model_index]
        else:
            self.nonlinearity_name = \
                config.get(["model", "student_nonlinearity"])

        if self.nonlinearity_name == 'relu':
            return F.relu
        elif self.nonlinearity_name == 'sigmoid':
            return torch.sigmoid
        elif self.nonlinearity_name == 'scaled_erf':
            return lambda x: torch.erf(x / math.sqrt(2))
        elif self.nonlinearity_name == 'linear':
            return linear_function
        elif self.nonlinearity_name == 'leaky_relu':
            return F.leaky_relu
        else:
            raise ValueError("Unknown non-linearity")

    def _construct_layers(self) -> None:
        """
        This method instantiates layers (input, hidden and output) according to
        dimensions specified in configuration. Note this method makes a call to
        the abstract _construct_output_layers, which is implemented by the
        child.
        """
        self.layers = nn.ModuleList([])

        input_layer = nn.Linear(
            self.input_dimension, self.hidden_dimensions[0], bias=self.bias
            )
        self._initialise_weights(input_layer)
        self.layers.append(input_layer)

        for h in range(len(self.hidden_dimensions[:-1])):
            hidden_layer = nn.Linear(
                self.hidden_dimensions[h], self.hidden_dimensions[h + 1],
                bias=self.bias
                )
            self._initialise_weights(hidden_layer)
            self.layers.append(hidden_layer)

        if 'teacher' in self.model_type and self.normalise_teachers:
            for i, layer in enumerate(self.layers):
                with torch.no_grad():
                    layer_weights = layer.state_dict()['weight']
                    layer_norm = torch.norm(layer_weights)
                    normalised_layer = np.sqrt(self.input_dimension) * (layer_weights / layer_norm)
                    self.layers[i].weight[0] = normalised_layer[0]

        self._construct_output_layers()

    @abstractmethod
    def _construct_output_layers(self):
        """This method instantiates the output layer."""
        raise NotImplementedError("Base class method")

    def _get_model_type(self) -> str:
        """returns model_type attribute"""
        return self.model_type

    def _initialise_weights(self, layer: nn.Module) -> None:
        """
        This method performs weight initialisation for a given layer.
        Initialisation is in place. The exact initialisation procedure
        is specified by the config.

        Args:
            layer: The layer to be initialised.
        """
        if self.initialisation_std is not None:
            torch.nn.init.normal_(layer.weight, std=self.initialisation_std)
            if self.bias:
                torch.nn.init.normal_(layer.bias, std=self.initialisation_std)
        if self.model_type == "student" and self.symmetric_student_initialisation:
            # copy first component of weight matrix to
            # others to ensure zero overlap
            base_parameters = copy.deepcopy(layer.state_dict()['weight'][0])
            for dim in range(1, len(layer.state_dict()['weight'])):
                layer.state_dict()['weight'][dim] = base_parameters

    def freeze_weights(self) -> None:
        """Freezes weights in graph (always called for teacher)"""
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method performs the forward pass. This implements the
        abstract method from the nn.Module base class.

        Args:
            x: Input tensor

        Returns:
            y: Output of network
        """
        for layer in self.layers:
            x = self.nonlinear_function(
                self.forward_scaling * layer(x)
                )

        y = self._output_forward(x)

        return y

    @abstractmethod
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")
