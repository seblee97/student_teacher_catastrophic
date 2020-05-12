from typing import List

from context import utils

class LoggingTemplate(utils._Template):

    LEVELS = ["logging"]
    OPTIONAL = []
    
    # Logging level fields
    VERBOSE = utils.Field(
        name="verbose", types=(bool), reqs=None
    )

    VERBOSE_TB = utils.Field(
        name="verbose_tb", types=(bool), reqs=None
    )

    CHECKPOINT_FREQUENCY = utils.Field(
        name="checkpoint_frequency", types=(int), reqs=None
    )
        
    LOG_TO_DF = utils.Field(
        name="log_to_df", types=(bool), reqs=None
    )

    MERGE_AT_CHECKPOINT = utils.Field(
        name="merge_at_checkpoint", types=(bool), reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.VERBOSE,
            cls.VERBOSE_TB,
            cls.CHECKPOINT_FREQUENCY,
            cls.LOG_TO_DF,
            cls.MERGE_AT_CHECKPOINT
        ]