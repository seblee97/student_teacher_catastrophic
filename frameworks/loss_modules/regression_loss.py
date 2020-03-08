from .base_loss import _BaseLoss 

from typing import Dict

class RegressionLoss(_BaseLoss):

    def __init__(self, config: Dict):
        _BaseLoss.__init__(self, config=config)

    # def compute_loss(self, prediction, target):
    #     return self.loss_function(prediction, target)