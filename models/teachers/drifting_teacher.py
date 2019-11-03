from .teacher import Teacher

import torch
import torch.distributions as tdist
import torch.nn as nn

from typing import Dict

class DriftingTeacher(Teacher):

    def __init__(self, config: Dict) -> None:

        Teacher.__init__(self, config=config)
        self.drift_distribution = tdist.Normal(torch.Tensor([0]), torch.Tensor([config.get(["task", "drift_size"])]))
        self.forward_count = 0
    
    def rotate_weights(self):
        for parameter_layer in self.state_dict():
            drift = self.drift_distribution.sample((len(self.state_dict()[parameter_layer]),))
            self.state_dict()[parameter_layer] = self.state_dict()[parameter_layer] + drift