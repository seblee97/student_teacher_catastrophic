from typing import List

from utils import Parameters, _Template, Field
from .task_template import TaskTemplate 
from .training_template import TrainingTemplate 
from .testing_template import TestingTemplate
from .model_template import ModelTemplate
from .curriculum_template import CurriculumTemplate
from .teachers_template import TeachersTemplate
from .logging_template import LoggingTemplate
from .data_template import DataTemplate

class ConfigTemplate(_Template):

    LEVELS = "ROOT"
    OPTIONAL: List[str] = ["drift_teachers", "mnist_data", "pure_mnist", "trained_mnist"]

    # root level fields
    EXPERIMENT_NAME = Field(
        name="experiment_name", types=[type(None), str], reqs=None
    )

    USE_GPU = Field(
        name="use_gpu", types=[bool], reqs=None
    )

    SEED = Field(
        name="seed", types=[int], reqs=None
    )

    TASK_TEMPLATE = TaskTemplate
    TRAINING_TEMPLATE = TrainingTemplate
    DATA_TEMPLATE = DataTemplate
    TESTING_TEMPLATE = TestingTemplate
    LOGGING_TEMPLATE = LoggingTemplate
    MODEL_TEMPLATE = ModelTemplate
    CURRICULUM_TEMPLATE = CurriculumTemplate
    TEACHERS_TEMPLATE = TeachersTemplate

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.EXPERIMENT_NAME,
            cls.USE_GPU,
            cls.SEED,

            cls.TASK_TEMPLATE,
            cls.TRAINING_TEMPLATE,
            cls.DATA_TEMPLATE,
            cls.TESTING_TEMPLATE,
            cls.LOGGING_TEMPLATE,
            cls.MODEL_TEMPLATE,
            cls.CURRICULUM_TEMPLATE,
            cls.TEACHERS_TEMPLATE
        ]
