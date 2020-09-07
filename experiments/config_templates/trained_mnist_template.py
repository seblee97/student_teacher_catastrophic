from typing import List

from utils import _Template, Field


class TrainedMNISTTemplate(_Template):

    LEVELS = ["trained_mnist"]
    OPTIONAL: List[str] = []

    SAVE_WEIGHT_PATH = Field(name="save_weight_path", types=[str], reqs=None)

    CONVERGENCE_CRITERION = Field(name="convergence_criterion", types=[float], reqs=None)

    LEARNING_RATE = Field(name="learning_rate", types=[float, int], reqs=None)

    OUTPUT_DIMENSION = Field(name="output_dimension", types=[int], reqs=None)

    BATCH_SIZE = Field(name="batch_size", types=[int], reqs=None)

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.SAVE_WEIGHT_PATH, cls.CONVERGENCE_CRITERION, cls.LEARNING_RATE,
            cls.OUTPUT_DIMENSION, cls.BATCH_SIZE
        ]
