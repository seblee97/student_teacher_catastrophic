from typing import List

import numpy as np

from utils import Field
from utils import _Template


class TeachersTemplate(_Template):

    LEVELS = ["teachers"]
    OPTIONAL: List[str] = []

    # Teachers level fields
    OVERLAP_PERCENTAGES = Field(
        name="overlap_percentages",
        types=[list],
        reqs=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)])

    OVERLAP_ROTATIONS = Field(
        name="overlap_rotations",
        types=[list],
        reqs=[
            lambda x: all((isinstance(y, float) or isinstance(y, int)) and y >= 0 and y <= 2 * np.pi
                          for y in x)
        ])

    TEACHER_NOISE = Field(
        name="teacher_noise",
        types=[list],
        reqs=[lambda x: all(isinstance(y, (float, int)) and y >= 0 for y in x)])

    OVERLAP_TYPE = Field(
        name="overlap_type", types=[str], reqs=[lambda x: x in ["rotation", "copy"]])

    @classmethod
    def get_fields(cls) -> List:
        return [cls.OVERLAP_PERCENTAGES, cls.OVERLAP_ROTATIONS, cls.TEACHER_NOISE, cls.OVERLAP_TYPE]
