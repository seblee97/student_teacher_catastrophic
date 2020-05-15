from utils import _Template, Field

from typing import List


class TeachersTemplate(_Template):

    LEVELS = ["teachers"]
    OPTIONAL: List[str] = []

    # Teachers level fields
    OVERLAP_PERCENTAGES = Field(
        name="overlap_percentages", types=[list],
        reqs=[lambda x: all(isinstance(y, int) and y >= 0 for y in x)]
    )

    TEACHER_NOISE = Field(
        name="teacher_noise", types=[list],
        reqs=[lambda x: all(isinstance(y, (float, int)) and y >= 0 for y in x)]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.OVERLAP_PERCENTAGES,
            cls.TEACHER_NOISE
        ]
