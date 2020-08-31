from typing import List

from utils import _Template, Field


class LoggingTemplate(_Template):

    LEVELS = ["logging"]
    OPTIONAL: List[str] = []

    # Logging level fields
    VERBOSE = Field(name="verbose", types=[bool], reqs=None)

    VERBOSE_TB = Field(name="verbose_tb", types=[int], reqs=[lambda x: x in [0, 1, 2]])

    CHECKPOINT_FREQUENCY = Field(name="checkpoint_frequency", types=[int], reqs=None)

    LOG_TO_DF = Field(name="log_to_df", types=[bool], reqs=None)

    MERGE_AT_CHECKPOINT = Field(name="merge_at_checkpoint", types=[bool], reqs=None)

    SAVE_WEIGHTS_AT_SWITCH = Field(name="save_weights_at_switch", types=[bool], reqs=None)

    SAVE_INITIAL_WEIGHTS = Field(name="save_initial_weights", types=[bool], reqs=None)

    LOG_OVERLAPS = Field(name="log_overlaps", types=[bool], reqs=None)

    SAVE_WEIGHT_PERIOD = Field(name="save_weight_period", types=[int], reqs=None)

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.VERBOSE, cls.VERBOSE_TB, cls.CHECKPOINT_FREQUENCY, cls.LOG_TO_DF,
            cls.MERGE_AT_CHECKPOINT, cls.SAVE_WEIGHTS_AT_SWITCH, cls.SAVE_INITIAL_WEIGHTS,
            cls.LOG_OVERLAPS, cls.SAVE_WEIGHT_PERIOD
        ]
