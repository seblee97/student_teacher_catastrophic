from context import utils

from typing import List

class TrainingTemplate(utils._Template):

    LEVELS = ["training"]
    OPTIONAL = []

    # Training level fields
    TOTAL_TRAINING_STEPS = utils.Field(
        name="total_training_steps", types=(type(None), int), reqs=[lambda x: x is None or x > 0]
    )

    TRAIN_BATCH_SIZE = utils.Field(
        name="train_batch_size", types=(int), reqs=[lambda x: x > 0]
    )
    
    TEST_BATCH_SIZE = utils.Field(
        name="test_batch_size", types=(int), reqs=[lambda x: x > 0]
    )

    LEARNING_RATE = utils.Field(
        name="learning_rate", types=(float, int), reqs=[lambda x: x > 0]
    )

    WEIGHT_DECAY = utils.Field(
        name="weight_decay", types=(float, int), reqs=[lambda x: x >= 0]
    )

    LOSS_FUNCTION = utils.Field(
        name="loss_function", types=(str), reqs=[lambda x: x in ["mse", "bce"]]
    )

    INPUT_SOURCE = utils.Field(
        name="input_source", types=(str), reqs=[lambda x: x in ["mnist", "iid_gaussian"]]
    )

    SCALE_OUTPUT_BACKWARD = utils.Field(
        name="scale_output_backward", types=(bool), reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.TOTAL_TRAINING_STEPS,
            cls.TRAIN_BATCH_SIZE,
            cls.TEST_BATCH_SIZE,
            cls.LEARNING_RATE,
            cls.WEIGHT_DECAY,
            cls.LOSS_FUNCTION,
            cls.INPUT_SOURCE,
            cls.SCALE_OUTPUT_BACKWARD
        ]