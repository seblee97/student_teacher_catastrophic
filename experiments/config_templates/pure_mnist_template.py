from typing import List

from utils import _Template, Field


class PureMNISTTemplate(_Template):

    LEVELS = ["pure_mnist"]
    OPTIONAL: List[str] = []

    TEACHER_DIGITS = Field(
        name="teacher_digits", types=[list],
        reqs=[
            lambda x: all(isinstance(y, list) for y in x),
            lambda x: all(
                all(isinstance(z, int) and 0 <= z <= 9 for z in y) for y in x
                )
            ]
    )

    ROTATIONS = Field(
        name="rotations", types=[list], reqs=[
            lambda x: all(isinstance(y, list) for y in x),
            lambda x: all(
                all(isinstance(z, int) and z in [0, 90, 180, 360]
                    for z in y
                    )
                for y in x)
            ]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.TEACHER_DIGITS,
            cls.ROTATIONS,
        ]
