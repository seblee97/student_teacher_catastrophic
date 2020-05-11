from typing import List

from context import utils
from .task_template import TaskTemplate 
from .training_template import TrainingTemplate 
from .testing_template import TestingTemplate
from .model_template import ModelTemplate
from .curriculum_template import CurriculumTemplate
from .teachers_template import TeachersTemplate

class ConfigTemplate(utils._Template):

    LEVELS = "ROOT"

    # root level fields
    EXPERIMENT_NAME = utils.Field(
        name="experiment_name", types=(type(None), str), reqs=None
    )

    USE_GPU = utils.Field(
        name="use_gpu", types=(bool), reqs=None
    )

    SEED = utils.Field(
        name="seed", types=(int), reqs=None
    )
        
    VERBOSE = utils.Field(
        name="verbose", types=(bool), reqs=None
    )

    VERBOSE_TB = utils.Field(
        name="verbose_tb", types=(bool), reqs=None
    )

    CHECKPOINT_FREQUENCY = utils.Field(
        name="checkpoint_frequency", types=(int), reqs=None
    )
        
    LOG_TO_DF = utils.Field(
        name="log_to_df", types=(bool), reqs=None
    )

    TASK_TEMPLATE = TaskTemplate
    TRAINING_TEMPLATE = TrainingTemplate
    TESTING_TEMPLATE = TestingTemplate
    MODEL_TEMPLATE = ModelTemplate
    CURRICULUM_TEMPLATE = CurriculumTemplate
    TEACHERS_TEMPLATE = TeachersTemplate

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.EXPERIMENT_NAME,
            cls.USE_GPU,
            cls.SEED,
            cls.VERBOSE,
            cls.VERBOSE_TB,
            cls.CHECKPOINT_FREQUENCY,
            cls.LOG_TO_DF,
            cls.LOG_TO_DF,

            cls.TASK_TEMPLATE,
            cls.TRAINING_TEMPLATE,
            cls.TESTING_TEMPLATE,
            cls.MODEL_TEMPLATE,
            cls.CURRICULUM_TEMPLATE,
            cls.TEACHERS_TEMPLATE
        ]
