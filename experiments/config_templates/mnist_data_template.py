from typing import List

from context import utils

class MNISTDataTemplate(utils._Template):

    LEVELS = ["mnist_data"]
    OPTIONAL = []

    DATA_PATH = utils.Field(
        name="data_path", types=(str), reqs=None
    )

    PCA_INPUT = utils.Field(
        name="pca_input", types=(int), reqs=[lambda x: x >= 0 or x == -1]
    )

    STANDARDISE = utils.Field(
        name="standardise", types=(bool), reqs=None
    )

    NOISE = utils.Field(
        name="noise", types=(type(None), float, int), reqs=[lambda x: x is None or x > 0]
    )
  
    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.DATA_PATH, 
            cls.PCA_INPUT,
            cls.STANDARDISE,
            cls.NOISE,
        ]
        