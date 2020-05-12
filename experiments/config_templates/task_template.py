from context import utils

from typing import List

class TaskTemplate(utils._Template):

    LEVELS = ["task"]
    OPTIONAL = []

    # Task level fields
    LABEL_TASK_BOUNDARIES = utils.Field(
        name="label_task_boundaries", types=(bool), reqs=None
    )

    LEARNER_CONFIGURATION = utils.Field(
        name="learner_configuration", types=(str), reqs=[lambda x: x in ["continual", "meta"]]
    )
    
    TEACHER_CONFIGURATION = utils.Field(
        name="teacher_configuration", types=(str), reqs=[lambda x: x in ["overlapping", "mnist", "trained_mnist"]]
    )

    NUM_TEACHERS = utils.Field(
        name="num_teachers", types=(int), reqs=None
    )

    LOSS_TYPE = utils.Field(
        name="loss_type", types=(str), reqs=[lambda x: x in ["regression", "classification"]]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.LABEL_TASK_BOUNDARIES,
            cls.LEARNER_CONFIGURATION,
            cls.TEACHER_CONFIGURATION,
            cls.NUM_TEACHERS,
            cls.LOSS_TYPE
        ]