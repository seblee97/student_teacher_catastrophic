from typing import List

from utils import _Template


class TrainedMNISTTemplate(_Template):

    OPTIONAL: List[str] = []

    @classmethod
    def get_fields(cls) -> List:
        return [
        ]
