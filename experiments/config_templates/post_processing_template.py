from typing import List

from utils import _Template, Field


class PostProcessingTemplate(_Template):

    LEVELS = ["post_processing"]
    OPTIONAL: List[str] = []

    COMBINE_PLOTS = Field(
        name="combine_plots", types=[bool], reqs=None
    )

    SHOW_LEGENDS = Field(
        name="show_legends", types=[bool], reqs=None
    )

    CROP_X = Field(
        name="crop_x",
        types=[
            type(None), (float, int), (float, float),
            (int, float), (int, int)
            ],
        reqs=[
            lambda x: x is None
            or (
                x[1] > x[0]
                and
                x[1] < 1
                and
                x[0] >= 0
                )]
    )

    PLOT_HEIGHT = Field(
        name="plot_height", types=[int, float], reqs=[lambda x: x > 0]
    )

    PLOT_WIDTH = Field(
        name="plot_width", types=[int, float], reqs=[lambda x: x > 0]
    )

    @classmethod
    def get_fields(cls) -> List:
        return [
            cls.COMBINE_PLOTS,
            cls.SHOW_LEGENDS,
            cls.CROP_X,
            cls.PLOT_HEIGHT,
            cls.PLOT_WIDTH
        ]
