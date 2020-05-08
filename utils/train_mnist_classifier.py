from models.networks.base_network import Model
from components.data_modules import PureMNISTData

import torch
import torch.nn as nn 

import numpy as np
from collections import deque

from typing import Dict

class ClassificationTeacher(Model):

    """Classification - threshold output"""

    def __init__(self, config: Dict) -> None:

        self._output_dimension = config.get(["trained_mnist", "output_dimension"])
        Model.__init__(self, config=config, model_type="teacher_0")

    def _construct_output_layers(self):
        self.output_layer = self._initialise_weights(nn.Linear(self.hidden_dimensions[-1], self._output_dimension, bias=self.bias))

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override teacher output forward, add sign output
        """
        y = self.output_layer(x)
        
        sigmoid_y = torch.sigmoid(y)
        # negative_class_probabilities = 1 - sigmoid_y

        return sigmoid_y
        
        # print(negative_class_probabilities)
        # return torch.cat((negative_class_probabilities, sigmoid_y), dim=1)
        # log_softmax = torch.log(torch.cat((negative_class_probabilities, sigmoid_y), dim=1))
        # # import pdb; pdb.set_trace()
        # return log_softmax

class MNISTTrainer:

    def __init__(self, config: Dict, task_index: int):

        self.task_index=task_index
        
        self.batch_size = config.get(["trained_mnist", "batch_size"])

        self.model = ClassificationTeacher(config=config)
        self.data_module = PureMNISTData(config, override_batch_size=self.batch_size)
        self.data_module.signal_task_boundary_to_data_generator(new_task=task_index)

        self.convergence_criterion = config.get(["trained_mnist", "convergence_criterion"])
        self.lr = config.get(["trained_mnist", "learning_rate"])

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_function = nn.BCELoss()

    def train(self):

        running_mean_loss = deque([np.inf], maxlen=10)

        # import pdb; pdb.set_trace()
        b = 0
        s = 0

        while np.mean(running_mean_loss) > self.convergence_criterion:

            b += 32
            s += 1

            # if s == 370:
            #     import pdb; pdb.set_trace()

            batch = self.data_module.get_batch() # returns dictionary 

            batch_input = batch.get('x')
            batch_labels = batch.get('y').type(torch.FloatTensor) # returns None unless using pure MNIST teachers

            predictions = self.model(batch_input).squeeze()

            self.optimiser.zero_grad()
            # import pdb; pdb.set_trace()
            loss = self.loss_function(predictions, batch_labels)
            loss.backward()
            self.optimiser.step()

            mean_loss = float(torch.mean(loss))
            running_mean_loss.append(mean_loss)

            print(b, s, np.mean(running_mean_loss), [w.grad for w in self.model.parameters()][0].sum())

    def save_model_weights(self, path: str):
        torch.save(self.model.state_dict(), path)
