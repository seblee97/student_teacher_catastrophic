from utils import _Template, Field

from typing import List


class CurriculumTemplate(_Template):

    LEVELS = ["curriculum"]
    OPTIONAL: List[str] = []

    # Curriculum level fields
    TYPE = Field(
        name="type", types=[str], reqs=[lambda x: x in ["custom", "standard"]]
    )

    SELECTION_TYPE = Field(
        name="selection_type", types=[str],
        reqs=[lambda x: x in ["random", "cyclical"]]
    )

    STOPPING_CONDITION = Field(
        name="stopping_condition", types=[str],
        reqs=[lambda x: x in [
            "fixed_period", "single_threshold", "threshold_sequence"
            ]]
    )

    FIXED_PERIOD = Field(
        name="fixed_period", types=[int], reqs=[lambda x: x > 0]
    )

    LOSS_THRESHOLD = Field(
        name="loss_threshold", types=[list, float],
        reqs=[
            lambda x:
            (isinstance(x, float) and x > 0)
            or (
                isinstance(x, list) and
                all(isinstance(y, float) and y > 0 for y in x)
                )
            ]
    )

    CUSTOM = Field(
        name="custom", types=[list],
        reqs=[lambda x: all(isinstance(y, int) for y in x)]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.TYPE,
            cls.SELECTION_TYPE,
            cls.STOPPING_CONDITION,
            cls.FIXED_PERIOD,
            cls.LOSS_THRESHOLD,
            cls.CUSTOM
        ]
