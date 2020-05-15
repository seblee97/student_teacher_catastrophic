from abc import ABC

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from utils import Parameters


class _BaseLoss(ABC):

    def __init__(self, config: Parameters):

        config_loss = config.get(["training", "loss_function"])
        self._setup_loss_function(config_loss)

    def _setup_loss_function(self, loss_name: str) -> None:
        """instantiate torch loss function"""
        self.loss_function: _Loss
        if loss_name == 'mse':
            self.loss_function = nn.MSELoss()
        elif loss_name == 'bce':
            self.loss_function = nn.BCELoss()
        elif loss_name == 'l1':
            self.loss_function = nn.L1Loss()
        elif loss_name == 'smooth_l1':
            self.loss_function = nn.SmoothL1Loss()
        else:
            raise NotImplementedError(
                "{} is not currently supported, \
                    please use mse loss or cross_entropy \
                    for mnist".format(loss_name)
                )

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates loss of prediction of student vs. target from teacher
        Loss function determined by configuration

        :param prediction: prediction made by student network on given input
        :param target: target - teacher output on same input

        :return loss: loss between target (from teacher) and prediction
        (from student)
        """
        loss = 0.5 * self.loss_function(prediction, target)
        return loss
