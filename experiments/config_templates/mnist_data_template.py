from typing import List

from utils import _Template, Field


class MNISTDataTemplate(_Template):

    LEVELS = ["mnist_data"]
    OPTIONAL: List[str] = []

    DATA_PATH = Field(
        name="data_path", types=[str], reqs=None
    )

    PCA_INPUT = Field(
        name="pca_input", types=[int], reqs=[lambda x: x >= 0 or x == -1]
    )

    STANDARDISE = Field(
        name="standardise", types=[bool], reqs=None
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.DATA_PATH,
            cls.PCA_INPUT,
            cls.STANDARDISE,
        ]
