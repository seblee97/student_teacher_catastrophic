import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim

from typing import List, Tuple

class Model(nn.Module):

    def __init__(
        self, input_dimension: int, hidden_dimensions: List[int], 
        output_dimension: int, nonlinearity: str, add_noise: bool=False,
        soft_committee: bool=True
        ):
        super(Model, self).__init__()

        self.nonlinearity = nonlinearity
        self.layers = nn.ModuleList([])
        
        input_layer = nn.Linear(input_dimension, hidden_dimensions[0])
        self.layers.append(input_layer)

        for h in hidden_dimensions[:-1]:
            hidden_layer = nn.Linear(hidden_dimensions[h], hidden_dimensions[h + 1])
            self.layers.append(hidden_layer)

        output_layer = nn.Linear(hidden_dimensions[-1], output_dimension)
        if soft_committee:
            for param in output_layer.parameters():
                param.requires_grad = False
        self.layers.append(output_layer)

        self.add_noise = add_noise

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self._initialise_weights()

    def _initialise_weights(self):
        for layer in self.layers:
            if self.nonlinearity == 'relu':
                std = 1 / np.sqrt(self.input_dimension)
                torch.nn.init.normal_(layer.weight, std=std)
                torch.nn.init.normal_(layer.bias, std=std)

            elif self.nonlinearity == 'sigmoid' or 'linear':
                torch.nn.init.normal_(layer.weight)
                torch.nn.init.normal_(layer.bias)

    def forward(self, x):

        if self.nonlinearity == 'relu':
            nonlinear_function = F.relu
        elif self.nonlinearity == 'sigmoid':
            nonlinear_function = torch.sigmoid
        else:
            raise ValueError("Unknown linearity. Please use 'relu' or 'sigmoid'")

        for layer in self.layers:
            x = nonlinear_function(layer(x))

        if self.add_noise:
            noise = torch.randn((self.output_dimension))
            x = x + noise

        return x

class StudentTeacher:

    def __init__(
        self, num_teachers: int, input_dimension: int, teacher_hidden: List[int], 
        student_hidden: List[int], output_dimension: int, nonlinearity: str,
        curriculum: List[Tuple[int, int]], learning_rate: float=1e-4, 
        add_teacher_noise: bool=True, soft_committee: bool=True
        ):
        """
        :param num_teachers: the number of teachers to initialise
        :param input_dimension: the length of the input into both student and teacher networks
        :param teacher_hidden: list of integers denoting number of hidden units in each layer of teacher
        :param student_hidden: list of integers denoting number of hidden units in each layer of students
        :param output_dimension: the length of the output of both student and teacher networks
        :param curriculum: list of teacher indices and steps on which to train on them 
                           e.g. [(0, 100), (2, 150)]  will train on teacher 0 for 100 steps followed by teacher 3 for 150 steps.
        :param learning_rate: parameter modulating length of step to take for optimiser
        """
        self.student_network = Model(
            input_dimension=input_dimension, hidden_dimensions=student_hidden, 
            output_dimension=output_dimension, nonlinearity=nonlinearity,
            soft_committee=soft_committee
            )

        self.teachers = []
        for _ in range(num_teachers):
            teacher = Model(
                input_dimension=input_dimension, hidden_dimensions=teacher_hidden, 
                output_dimension=output_dimension, nonlinearity=nonlinearity, 
                add_noise=add_teacher_noise, soft_committee=soft_committee
                ) 
            self.teachers.append(teacher)

        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension   
        
        self.curriculum = curriculum
        
        trainable_parameters = filter(lambda param: param.requires_grad, self.student_network.parameters())
        self.optimiser = optim.SGD(trainable_parameters, lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def train(self):

        losses = []

        for teacher_index, steps in self.curriculum:
            for s in range(steps):
                random_input = torch.randn(self.input_dimension)
                teacher_output = self.teachers[teacher_index](random_input)
                student_output = self.student_network(random_input)

                # import pdb; pdb.set_trace()

                self.optimiser.zero_grad()

                loss = self.compute_loss(student_output, teacher_output)
                losses.append(float(loss))

                loss.backward()

                self.optimiser.step()

        return losses

    def compute_loss(self, prediction, target):
        # import pdb; pdb.set_trace()
        loss = self.loss_function(prediction, target)
        return loss


        
