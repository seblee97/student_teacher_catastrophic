from typing import List

from utils import _Template, Field


class IIDDataTemplate(_Template):

    LEVELS = ["iid_data"]
    OPTIONAL: List[str] = []

    DATASET_SIZE = Field(
        name="dataset_size", types=[int, str],
        reqs=[lambda x: x == "inf" or (isinstance(x, int) and x > 0)]
    )

    MEAN = Field(
        name="mean", types=[int, float], reqs=None
    )

    VARIANCE = Field(
        name="variance", types=[int, float], reqs=[lambda x: x > 0]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.DATASET_SIZE,
            cls.MEAN,
            cls.VARIANCE
        ]
