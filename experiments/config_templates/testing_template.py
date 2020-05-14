from utils import _Template, Field

from typing import List

class TestingTemplate(_Template):

    LEVELS = ["testing"]
    OPTIONAL: List[str] = []

    # Testing level fields
    TEST_FREQUENCY = Field(
        name="test_frequency", types=[int], reqs=[lambda x: x > 0]
    )

    OVERLAP_FREQUENCY = Field(
        name="overlap_frequency", types=[int], reqs=[lambda x: x > 0]
    )
    
    TEST_ALL_TEACHERS = Field(
        name="test_all_teachers", types=[bool], reqs=None
    )
        
    TEST_BATCH_SIZE = Field(
        name="test_batch_size", types=[int], reqs=[lambda x: x > 0]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.TEST_FREQUENCY,
            cls.OVERLAP_FREQUENCY,
            cls.TEST_ALL_TEACHERS,
            cls.TEST_BATCH_SIZE
        ]