from typing import List

from utils import _Template, Field


class IIDDataTemplate(_Template):

    LEVELS = ["iid_data"]
    OPTIONAL: List[str] = []

    MEAN = Field(
        name="mean", types=[int, float], reqs=None
    )

    VARIANCE = Field(
        name="variance", types=[int, float], reqs=[lambda x: x > 0]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.MEAN,
            cls.VARIANCE
        ]
