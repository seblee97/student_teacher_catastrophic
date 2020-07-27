from utils import Parameters
from .base_loss import _BaseLoss


class RegressionLoss(_BaseLoss):

    def __init__(self, config: Parameters):
        _BaseLoss.__init__(self, config=config)
