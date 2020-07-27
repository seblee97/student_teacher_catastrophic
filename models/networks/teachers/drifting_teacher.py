import torch.distributions as tdist

from utils import Parameters
from .teacher import Teacher


class DriftingTeacher(Teacher):

    def __init__(self, config: Parameters, index: int) -> None:

        Teacher.__init__(self, config=config, index=index)
        self.drift_size = config.get(["task", "drift_size"])
        self.drift_distribution = tdist.Normal(0, self.drift_size)
        self.forward_count = 0

    def rotate_weights(self):
        for parameter_layer in self.state_dict():
            drift = self.drift_distribution.sample((len(self.state_dict()[parameter_layer]),))
            self.state_dict()[parameter_layer] \
                = self.state_dict()[parameter_layer] + drift
