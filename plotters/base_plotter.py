import abc
import os
from itertools import cycle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import constants
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


class BasePlotter(abc.ABC):
    """Base class for plotting generalisation errors and order parameters."""

    ERROR_IDENTIFIERS = {
        constants.Constants.GENERALISATION_ERROR: constants.Constants.GENERALISATION_ERROR_LABEL,
        constants.Constants.LOG_GENERALISATION_ERROR: constants.Constants.LOG_GENERALISATION_ERROR_LABEL,
    }

    OVERLAP_IDENTIFIERS = {
        constants.Constants.STUDENT_HEAD: constants.Constants.STUDENT_HEAD_LABEL,
        constants.Constants.STUDENT_SELF: constants.Constants.STUDENT_SELF_LABEL,
        constants.Constants.STUDENT_TEACHER_0: constants.Constants.STUDENT_TEACHER_0_LABEL,
        constants.Constants.STUDENT_TEACHER_1: constants.Constants.STUDENT_TEACHER_1_LABEL,
    }

    OTHER_IDENTIFIERS = {
        constants.Constants.TEACHER_INDEX: constants.Constants.TEACHER_INDEX 
    }

    IDENTIFIERS = {**ERROR_IDENTIFIERS, **OVERLAP_IDENTIFIERS, **OTHER_IDENTIFIERS}

    GRAPH_LAYOUT = (3, 3)

    # base range of colors
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # additional set of colors
    COLORS.extend(list(mcolors.CSS4_COLORS.values()))

    def __init__(
        self,
        save_folder: str,
        num_steps: int,
        log_overlaps: bool,
    ) -> None:
        """Class constructor.

        Args:
            save_folder: path to folder for saving plots.
            num_steps: total number of steps in the training run (used for scaling axes).
            log_overlaps: whether or not to plot overlaps (or just errors).
        """
        self._save_folder = save_folder
        self._num_steps = num_steps
        self._log_overlaps = log_overlaps

        self._setup_data()

    @abc.abstractmethod
    def _setup_data(self):
        """Setup data from relevant dataframes."""
        pass

    @abc.abstractmethod
    def make_plots(self) -> None:
        """Orchestration method for plotting ode logs, network logs, or both."""
        pass

    @abc.abstractmethod
    def _make_plot(
        self,
        data: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
        save_path: str,
    ) -> None:
        """Make plots for a set of results (e.g. ode or network or both).

        Args:
            dfs: mapping from type of results (ode, network etc.)
            to dataframes with results.
            save_path: path to save the plot.
        """
        pass

    def _collate_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """Group tags together according to identifiers.

        E.g. All error data columns begin with
        constants.Constants.GENERALISATION_ERROR, and are followed by an index.

        Args:
            tags: all tags (e.g. columns of a dataframe).

        Returns:
            tag_groups: dictionary of groups of tags.
        """
        tag_groups = {}

        # identifiers = list(self.ERROR_IDENTIFIERS.keys())
        # if self._log_overlaps:
        #     identifiers.extend(list(self.OVERLAP_IDENTIFIERS.keys()))

        for identifier in self.IDENTIFIERS.keys():
            id_tags = [tag for tag in tags if tag.startswith(identifier)]
            if id_tags:
                tag_groups[identifier] = id_tags

        return tag_groups

    def _get_figure_skeleton(
        self,
        height: Union[int, float],
        width: Union[int, float],
        num_columns: int,
        num_rows: int,
    ) -> Tuple[plt.figure, gridspec.GridSpec]:
        """Construct a figure skeleton with given number of columns, rows etc.

        Args:
            height: height of each sub figure.
            width: width of each sub figure.
            num_columns: number of figures per row.
            num_rows: number of figures per column.

        Returns:
            fig: matplotlib figure.
            spec: specification for figure layout
        """

        fig = plt.figure(
            constrained_layout=False, figsize=(num_columns * width, num_rows * height)
        )

        heights = [height for _ in range(num_rows)]
        widths = [width for _ in range(num_columns)]

        spec = gridspec.GridSpec(
            nrows=num_rows,
            ncols=num_columns,
            width_ratios=widths,
            height_ratios=heights,
        )

        return fig, spec

    def _plot_scalar(
        self,
        fig: plt.figure,
        spec: gridspec.GridSpec,
        row: int,
        col: int,
        tag_group_name: str,
        data_collection: Dict[str, Dict[str, np.ndarray]],
    ) -> plt.figure:
        """Add a sub plot to the figure.

        Args:
            fig: Overall matplotlip figure with plots.
            row: row index at which to add subplot.
            col: column index at which to add subplot.

        Returns:
            fig: Overall matplotlib figure, now populated with additional subplot.
        """
        fig_sub = fig.add_subplot(spec[row, col])

        # create cyclical iterator to ensure sufficient colors
        color_cycle = cycle(self.COLORS)

        color_dict = {}

        for key in list(data_collection.values())[0].keys():
            color_dict[key] = next(color_cycle)

        for data_type, data in data_collection.items():
            if data_type == constants.Constants.ODE:
                linestyle = constants.Constants.SOLID
            elif data_type == constants.Constants.SIM:
                linestyle = constants.Constants.DASHED
            for key, plot_data in data.items():
                data_indexing = "-".join(
                    [s for s in key.split(tag_group_name)[1].split("_") if s.isdigit()]
                )
                scaling = self._num_steps / len(plot_data)
                fig_sub.plot(
                    scaling * np.arange(len(plot_data)),
                    plot_data,
                    label=f"{self.IDENTIFIERS[tag_group_name]} {data_indexing} {data_type}",
                    color=color_dict[key],
                    linestyle=linestyle,
                )

        # labelling
        fig_sub.set_xlabel(constants.Constants.STEP)
        fig_sub.set_ylabel(self.IDENTIFIERS[tag_group_name])

        if len(color_dict) < 10:
            fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which="major", linestyle="-", linewidth="0.5", color="red", alpha=0.2
        )
        fig_sub.grid(
            which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.4
        )

        return fig
