from utils import _Template, Field

from typing import List


class TaskTemplate(_Template):

    LEVELS = ["task"]
    OPTIONAL: List[str] = []

    # Task level fields
    LABEL_TASK_BOUNDARIES = Field(
        name="label_task_boundaries", types=[bool], reqs=None
    )

    LEARNER_CONFIGURATION = Field(
        name="learner_configuration", types=[str],
        reqs=[lambda x: x in ["continual", "meta"]]
    )

    TEACHER_CONFIGURATION = Field(
        name="teacher_configuration", types=[str],
        reqs=[lambda x: x in ["overlapping", "pure_mnist", "trained_mnist"]]
    )

    NUM_TEACHERS = Field(
        name="num_teachers", types=[int], reqs=None
    )

    LOSS_TYPE = Field(
        name="loss_type", types=[str],
        reqs=[lambda x: x in ["regression", "classification"]]
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
