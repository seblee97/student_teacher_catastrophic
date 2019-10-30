import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import torch.distributions as tdist

import numpy as np
import copy

from typing import List, Tuple, Generator, Dict

from abc import ABC, abstractmethod

class Model(nn.Module):

    def __init__(self, config: Dict, model_type: str) -> None:
        """
        Multi-layer non-linear neural network class. For use in student-teacher framework.

        :param config: dictionary containing parameters to specify network configuration
        :param model_type: "teacher" or "student"
        """
        self.model_type = model_type # 'teacher' or 'student'

        assert self.model_type == 'teacher' or 'student', "Unknown model type {} provided to network".format(self.model_type)

        # extract relevant parameters from config
        self._extract_parameters(config=config)

        # initialise specified nonlinearity function
        self.nonlinearity_name = config.get(["model", "nonlinearity"])
        if self.nonlinearity_name == 'relu':
            self.nonlinear_function = F.relu
        elif self.nonlinearity_name == 'sigmoid':
            self.nonlinear_function = torch.sigmoid
        else:
            raise ValueError("Unknown non-linearity. Please use 'relu' or 'sigmoid'")

        super(Model, self).__init__()

        self._construct_layers()

    def _extract_parameters(self, config: Dict):
        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        self.hidden_dimensions = config.get(["model", "{}_hidden_layers".format(self.model_type)])
        self.initialisation_std = config.get(["model", "{}_initialisation_std".format(self.model_type)])
        self.add_noise = config.get(["model", "{}_add_noise".format(self.model_type)])
        self.bias = config.get(["model", "bias_parameters"])
        self.soft_committee = config.get(["model", "soft_committee"])
        self.task_setting = config.get(["task", "task_setting"])
        self.num_teachers = config.get(["task", "num_teachers"])

    def _construct_layers(self) -> None:
        """
        initiates layers (input, hidden and output) according to dimensions specified in configuration
        """
        self.layers = nn.ModuleList([])
        
        input_layer = nn.Linear(self.input_dimension, self.hidden_dimensions[0], bias=self.bias)
        self._initialise_weights(input_layer)
        self.layers.append(input_layer)

        for h in self.hidden_dimensions[:-1]:
            hidden_layer = nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1], bias=self.bias)
            self._initialise_weights(hidden_layer)
            self.layers.append(hidden_layer)

        self._construct_output_layers()

    @abstractmethod
    def _construct_output_layers(self):
        """
        initiates output layers in particular. Student may have different heads, scm has frozen output etc.
        """
        raise NotImplementedError("Base class method")

    def _get_model_type(self) -> str:
        """
        returns class attribute 'model type' i.e. 'teacher' or 'student'
        """
        return self.model_type

    def _initialise_weights(self, layer) -> None:
        """
        Weight initialisation method for given layer
        """
        if self.nonlinearity_name == 'relu':
            # std = 1 / np.sqrt(self.input_dimension)
            torch.nn.init.normal_(layer.weight, std=self.initialisation_std)
            if self.bias:
                torch.nn.init.normal_(layer.bias, std=self.initialisation_std)

        elif self.nonlinearity_name == 'sigmoid' or 'linear':
            torch.nn.init.normal_(layer.weight, std=self.initialisation_std)
            if self.bias:
                torch.nn.init.normal_(layer.bias, std=self.initialisation_std)

    def freeze_weights(self) -> None:
        """
        Freezes weights in graph (always called for teacher)
        """
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        :param x: input tensor to network
        :return y: output of network
        """
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))

        y = self._output_forward(x)

        if self.model_type == 'teacher' and self.noise_distribution:
            noise = self.noise_distribution.sample((self.batch_dimension,))
            y = y + noise

        return y

    @abstractmethod
    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class method")


class Student(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='student')

    def _construct_output_layers(self):

        if self.task_setting == 'continual':
            # create one head per teacher
            self.heads = nn.ModuleList([])
            for _ in range(self.num_teachers):
                output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
                self._initialise_weights(output_layer)
                if self.soft_committee:
                    for param in output_layer.parameters():
                        param.requires_grad = False
                self.heads.append(output_layer)
        
        elif self.task_setting == 'meta':
            self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
            self._initialise_weights(self.output_layer)
            if self.soft_committee:
                for param in self.output_layer.parameters():
                    param.requires_grad = False

        else:
            raise ValueError("Task setting {} not recognised. Please use 'meta' or 'continual'".format(self.task_setting))

    def set_task(self, task_index: int):
        self._current_teacher = task_index

    def test_all_tasks(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))
        if self.task_setting == 'continual': 
            task_outputs = [self.heads[t](x) for t in range(self.num_teachers)]
        elif self.task_setting == 'meta':
            task_outputs = self.output_layer(x)
        return task_outputs

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_setting == 'continual':
            y = self.heads[self._current_teacher](x)
        elif self.task_setting == 'meta':
            y = self.output_layer(x)
        return y

class Teacher(Model):

    def __init__(self, config: Dict) -> None:

        Model.__init__(self, config=config, model_type='teacher')

    def set_noise_distribution(self, mean: float, std: float):
        """
        Sets a normal distribution from which to sample noise that is added to output
        """
        if mean is None:
            self.noise_distribution = None
        else:
            self.noise_distribution = tdist.Normal(torch.Tensor([mean]), torch.Tensor([std]))

    def _construct_output_layers(self):
        self.output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        self._initialise_weights(self.output_layer)
        if self.soft_committee:
            for param in self.output_layer.parameters():
                param.requires_grad = False

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.output_layer(x)
        return y
