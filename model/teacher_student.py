import numpy as np 
import copy

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from typing import List, Tuple

class Model(nn.Module):

    def __init__(self, config, model_type: str):
        
        self.model_type = model_type # 'teacher' or 'student'

        # extract relevant parameters from config
        self.input_dimension = config.get("input_dimension")
        self.output_dimension = config.get("output_dimension")
        self.hidden_dimensions = config.get("{}_hidden_layers".format(self.model_type))
        self.initialisation_variance = config.get("{}_initialisation_variance".format(self.model_type))
        self.bias = config.get("bias")

        # initialise specified nonlinearity function
        nonlinearity_name = config.get("nonlinearity")
        if nonlinearity_name == 'relu':
            self.nonlinear_function = F.relu
        elif nonlinearity_name == 'sigmoid':
            self.nonlinear_function = torch.sigmoid
        else:
            raise ValueError("Unknown non-linearity. Please use 'relu' or 'sigmoid'")

        super(Model, self).__init__()

        self._construct_layers()

        self._initialise_weights()

    def _construct_layers(self):

        self.layers = nn.ModuleList([])
        
        input_layer = nn.Linear(self.input_dimension, self.hidden_dimensions[0], bias=self.bias)
        self.layers.append(input_layer)

        for h in hidden_dimensions[:-1]:
            hidden_layer = nn.Linear(self.hidden_dimensions[h], self.hidden_dimensions[h + 1], bias=self.bias)
            self.layers.append(hidden_layer)

        output_layer = nn.Linear(self.hidden_dimensions[-1], self.output_dimension, bias=self.bias)
        if soft_committee:
            for param in output_layer.parameters():
                param.requires_grad = False
        self.layers.append(output_layer)

    def _get_model_type(self):
        """
        returns class attribute 'model type' i.e. 'teacher' or 'student'
        """
        return self.model_type

    def _initialise_weights(self):
        for layer in self.layers:
            if self.nonlinearity == 'relu':
                # std = 1 / np.sqrt(self.input_dimension)
                torch.nn.init.normal_(layer.weight, std=self.initialisation_variance)
                # torch.nn.init.normal_(layer.bias, std=std)

            elif self.nonlinearity == 'sigmoid' or 'linear':
                torch.nn.init.normal_(layer.weight)
                # torch.nn.init.normal_(layer.bias)

    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.nonlinear_function(layer(x) / np.sqrt(self.input_dimension))

        y = self.layers[-1](x)

        if self.add_noise:
            noise = torch.randn(self.output_dimension)
            print('a', y)
            y = y + noise
            print('b', y)

        return y


class StudentTeacher:

    def __init__(self, config):
        """
        Experiment
        """

        # extract relevant parameters from config
        self.input_dimension = config.get("input_dimension")
        self.output_dimension = config.get("output_dimension")

        self.train_batch_size = config.get("train_batch_size")
        self.test_batch_size = config.get("test_batch_size")
        self.learning_rate = config.get("learning_rate")

        self.student_hidden = config.get("")
        self.teacher_hidden = config.get("")
        self.student_initialisation_variance = config.get("student_initialisation_variance")
        self.teacher_initialisation_variance = config.get("teacher_initialisation_variance")
        self.add_teacher_noise = config.get("add_teacher_noise")
        self.soft_committee = config.get("soft_committee")

        self.nonlinearity = config.get("nonlinearity")

        self.curriculum = config.get("curriculum")

        num_teachers = config.get("num_teachers")

        self.student_network = Model(config=config, model_type='student') 

        self.teachers = []
        for _ in range(num_teachers):
            teacher = Model(config=config, model_type='teacher')
            teacher.freeze_weights()
            self.teachers.append(teacher)
        
        trainable_parameters = filter(lambda param: param.requires_grad, self.student_network.parameters())
        
        self.optimiser = optim.SGD(trainable_parameters, lr=self.learning_rate)

        if config.get("loss") == 'mse':
            self.loss_function = nn.MSELoss()
        else:
            raise NotImplementedError("{} is not currently supported, please use mse loss".format(config.get("loss")))

        # generate fixed test data
        self.test_input_data = torch.randn(self.test_batch_size, self.input_dimension)
        self.test_teacher_outputs = [teacher(self.test_input_data) for teacher in self.teachers]

    def train(self):

        training_losses = []
        generalisation_errors = []

        for teacher_index, steps in self.curriculum:
            for s in range(steps):
                random_input = torch.randn(self.train_batch_size, self.input_dimension)
    
                teacher_output = self.teachers[teacher_index](random_input)
                student_output = self.student_network(random_input)

                # import pdb; pdb.set_trace()
                if s % 1000 == 0:
                    print(s)

                self.optimiser.zero_grad()

                loss = self.compute_loss(student_output, teacher_output)
                training_losses.append(float(loss))

                loss.backward()

                self.optimiser.step()

                generalisation_error = self.test(teacher_index)

                # print(float(generalisation_error))

                generalisation_errors.append(generalisation_error)

        return training_losses, generalisation_errors

    def test(self, teacher_index):
        student_outputs = [self.student_network(random_input) for random_input in self.test_input_data]
        # import pdb; pdb.set_trace()
        generalisation_error = np.mean([float(self.compute_loss(i, j)) for (i, j) in zip(student_outputs, self.test_teacher_outputs[teacher_index])])
        return generalisation_error

    def compute_loss(self, prediction, target):
        # import pdb; pdb.set_trace()
        loss = 0.5 * self.loss_function(prediction, target)
        return loss


        
