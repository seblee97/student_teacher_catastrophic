from typing import List

from context import utils

class MNISTTemplate(utils._Template):

    LEVELS = ["mnist"]

    DATA_PATH = utils.Field(
        name="data_path", types=(str), reqs=None
    )

    PCA_INPUT = utils.Field(
        name="pca_input", types=(int), reqs=[lambda x: x >= 0 or x == -1]
    )
  
    TEACHER_DIGITS = utils.Field(
        name="teacher_digits", types=(list), reqs=[
            lambda x: all(isinstance(y, list) for y in x), 
            lambda x: all(all(isinstance(z, int) and 0 <= z <= 9 for z in y) for y in x)
            ]
    )
  
    ROTATIONS = utils.Field(
        name="rotations", types=(list), reqs=[
            lambda x: all(isinstance(y, list) for y in x), 
            lambda x: all(all(isinstance(z, int) and z in [0, 90, 180, 360] for z in y) for y in x)
            ]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.DATA_PATH, 
            cls.PCA_INPUT,
            cls.TEACHER_DIGITS, 
            cls.ROTATIONS,
        ]
        