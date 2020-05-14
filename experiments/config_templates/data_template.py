from utils import _Template, Field

from typing import List

class DataTemplate(_Template):

    LEVELS: List[str] = ["data"]
    OPTIONAL: List[str] = []

    # Data level fields
    INPUT_SOURCE = Field(
        name="input_source", types=[str], reqs=[lambda x: x in ["mnist_stream", "mnist_digits", "even_greater", "iid_gaussian"]]
    )

    SAME_INPUT_DISTRIBUTION = Field(
        name="same_input_distribution", types=[bool], reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.INPUT_SOURCE,
            cls.SAME_INPUT_DISTRIBUTION
        ]