from utils import _Template, Field

from typing import List


class TrainingTemplate(_Template):

    LEVELS = ["training"]
    OPTIONAL: List[str] = []

    # Training level fields
    TOTAL_TRAINING_STEPS = Field(
        name="total_training_steps", types=[type(None), int], reqs=[lambda x: x is None or x > 0])

    TRAIN_BATCH_SIZE = Field(name="train_batch_size", types=[int], reqs=[lambda x: x > 0])

    LEARNING_RATE = Field(name="learning_rate", types=[float, int], reqs=[lambda x: x > 0])

    WEIGHT_DECAY = Field(name="weight_decay", types=[float, int], reqs=[lambda x: x >= 0])

    LOSS_FUNCTION = Field(name="loss_function", types=[str], reqs=[lambda x: x in ["mse", "bce"]])

    SCALE_HEAD_LR = Field(name="scale_head_lr", types=[bool], reqs=None)

    SCALE_HIDDEN_LR_FORWARD = Field(name="scale_hidden_lr_forward", types=[bool], reqs=None)

    SCALE_HIDDEN_LR_BACKWARD = Field(name="scale_hidden_lr_backward", types=[bool], reqs=None)

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.TOTAL_TRAINING_STEPS, cls.TRAIN_BATCH_SIZE, cls.LEARNING_RATE, cls.WEIGHT_DECAY,
            cls.LOSS_FUNCTION, cls.SCALE_HEAD_LR, cls.SCALE_HIDDEN_LR_FORWARD,
            cls.SCALE_HIDDEN_LR_BACKWARD
        ]
