from context import utils

from typing import List

class DataTemplate(utils._Template):

    LEVELS = ["data"]
    OPTIONAL = []

    # Data level fields
    INPUT_SOURCE = utils.Field(
        name="input_source", types=(str), reqs=[lambda x: x in ["mnist", "iid_gaussian"]]
    )

    SAME_INPUT_DISTRIBUTION = utils.Field(
        name="same_input_distribution", types=(bool), reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.INPUT_SOURCE,
            cls.SAME_INPUT_DISTRIBUTION
        ]