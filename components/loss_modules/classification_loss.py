from .base_loss import _BaseLoss 

from typing import Dict

class ClassificationLoss(_BaseLoss):

    def __init__(self, config: Dict):
        _BaseLoss.__init__(self, config=config)
