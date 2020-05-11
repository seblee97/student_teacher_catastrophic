from context import utils

from typing import List

class TeachersTemplate(utils._Template):

    LEVELS = ["teachers"]

    # Teachers level fields
    OVERLAP_PERCENTAGES = utils.Field(
        name="overlap_percentages", types=(list), reqs=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)]
    )

    TEACHER_NOISE = utils.Field(
        name="teacher_noise", types=(list), reqs=[lambda x: all(isinstance(y, (float, int)) and y >= 0 for y in x)]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.OVERLAP_PERCENTAGES,
            cls.TEACHER_NOISE
        ]