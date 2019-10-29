import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

import numpy as np

from typing import List, Tuple, Generator, Dict

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
        self.input_dimension = config.get(["model", "input_dimension"])
        self.output_dimension = config.get(["model", "output_dimension"])
        self.hidden_dimensions = config.get(["model", "{}_hidden_layers".format(self.model_type)])
        self.initialisation_std = config.get(["model", "{}_initialisation_std".format(self.model_type)])
        self.add_noise = config.get(["model", "{}_add_noise".format(self.model_type)])
        self.bias = config.get(["model", "bias_parameters"])
        self.soft_committee = config.get(["model", "soft_committee"])

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

        self._initialise_weights()

    def _construct_layers(self) -> None:
        """
        initiates layers (input, hidden and output) according to dimensions specified in configuration
        """
        self.layers = nn.ModuleList([])
        
        input_layer = nn.Linear(self.input_dimension, self.hidden_dimensions[0], bias=self.bias)
        self.layers.append(input_layer)

        for h in self.hidden_dimensions[:-1]:
            hidden_layer = nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1], bias=self.bias)
            self.layers.append(hidden_layer)

        output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        if self.soft_committee:
            for param in output_layer.parameters():
                param.requires_grad = False
        self.layers.append(output_layer)

    def _get_model_type(self) -> str:
        """
        returns class attribute 'model type' i.e. 'teacher' or 'student'
        """
        return self.model_type

    def _initialise_weights(self) -> None:
        """
        Weight initialisation method
        """
        for layer in self.layers:
            if self.nonlinearity_name == 'relu':
                # std = 1 / np.sqrt(self.input_dimension)
                torch.nn.init.normal_(layer.weight, std=self.initialisation_std)
                # torch.nn.init.normal_(layer.bias, std=std)

            elif self.nonlinearity_name == 'sigmoid' or 'linear':
                torch.nn.init.normal_(layer.weight, std=self.initialisation_std)
                # torch.nn.init.normal_(layer.bias)

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
        for layer in self.layers[:-1]:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))

        y = self.layers[-1](x)

        if self.add_noise:
            noise = torch.randn(self.output_dimension)
            y = y + noise

        return y