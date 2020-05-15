from models.networks.base_network import Model
from components.data_modules import MNISTDigitsData
from utils import Parameters

import torch
import torch.nn as nn

import numpy as np
from collections import deque

from typing import Dict


class ClassificationTeacher(Model):

    """Classification - threshold output"""

    def __init__(self, config: Parameters) -> None:

        self._output_dimension = \
            config.get(["trained_mnist", "output_dimension"])
        Model.__init__(self, config=config, model_type="teacher_0")

    def _construct_output_layers(self):
        self.output_layer = nn.Linear(
            self.hidden_dimensions[-1], self._output_dimension, bias=self.bias
            )
        self._initialise_weights(self.output_layer)

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override teacher output forward, add sign output
        """
        y = self.output_layer(x)

        sigmoid_y = torch.sigmoid(y)
        # negative_class_probabilities = 1 - sigmoid_y

        return sigmoid_y


class MNISTTrainer:

    def __init__(self, config: Parameters, task_index: int):

        self.task_index = task_index

        self.batch_size = config.get(["trained_mnist", "batch_size"])

        self.model = ClassificationTeacher(config=config)
        self.data_module = MNISTDigitsData(
            config, override_batch_size=self.batch_size
            )
        self.data_module.signal_task_boundary_to_data_generator(
            new_task=task_index
            )

        self.convergence_criterion = \
            config.get(["trained_mnist", "convergence_criterion"])
        self.lr = config.get(["trained_mnist", "learning_rate"])

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.BCELoss()

    def train(self):

        running_mean_loss = deque([np.inf], maxlen=10)

        while np.mean(running_mean_loss) > self.convergence_criterion:

            batch: Dict = self.data_module.get_batch()

            batch_input = batch.get('x')
            batch_labels = batch.get('y').type(torch.FloatTensor)

            predictions = self.model(batch_input).squeeze()

            self.optimiser.zero_grad()
            # import pdb; pdb.set_trace()
            loss = self.loss_function(predictions, batch_labels)
            loss.backward()
            self.optimiser.step()

            mean_loss = float(torch.mean(loss))
            running_mean_loss.append(mean_loss)

            print(
                np.mean(running_mean_loss),
                [w.grad for w in self.model.parameters()][0].sum()
                )

    def save_model_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
