from context import utils

from typing import List

class CurriculumTemplate(utils._Template):

    LEVELS = ["curriculum"]

    # Curriculum level fields
    TYPE = utils.Field(
        name="type", types=(str), reqs=[lambda x: x in ["custom", "standard"]]
    )

    SELECTION_TYPE = utils.Field(
        name="selection_type", types=(str), reqs=[lambda x: x in ["random", "cyclical"]]
    )
    
    STOPPING_CONDITION = utils.Field(
        name="stopping_condition", types=(str), reqs=[lambda x: x in ["fixed_period", "single_threshold", "threshold_sequence"]]
    )

    FIXED_PERIOD = utils.Field(
        name="fixed_period", types=(int), reqs=[lambda x: x > 0]
    )

    LOSS_THRESHOLD = utils.Field(
        name="loss_threshold", types=(list, float), reqs=[lambda x: (isinstance(x, float) and x > 0) or (isinstance(x, list) and all(isinstance(y, float) and y > 0 for y in x))]
    )

    CUSTOM = utils.Field(
        name="custom", types=(list), reqs=[lambda x: all(isinstance(y, int) for y in x)]
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