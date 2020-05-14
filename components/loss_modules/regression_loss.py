from .base_loss import _BaseLoss 
from utils import Parameters

from typing import Dict

class RegressionLoss(_BaseLoss):

    def __init__(self, config: Parameters):
        _BaseLoss.__init__(self, config=config)
        