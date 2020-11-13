import os
from itertools import cycle
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors

import constants


class Plotter:
    """Class for plotting generalisation error and overlaps."""

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

    IDENTIFIERS = {**ERROR_IDENTIFIERS, **OVERLAP_IDENTIFIERS}

    def __init__(
        self,
        save_folder: str,
        num_steps: int,
        log_overlaps: bool,
        ode_logger_path: Optional[str] = None,
        network_logger_path: Optional[str] = None,
    ):
        self._save_folder = save_folder
        self._num_steps = num_steps
        self._ode_logger_path = ode_logger_path
        self._network_logger_path = network_logger_path
        self._log_overlaps = log_overlaps

        if self._ode_logger_path is not None:
            self._ode_logger = pd.read_csv(ode_logger_path)
        if self._network_logger_path is not None:
            self._network_logger = pd.read_csv(network_logger_path)

    def make_plots(self):
        if self._ode_logger_path is not None:
            self._make_plot(
                dfs={constants.Constants.ODE: self._ode_logger},
                save_path=os.path.join(self._save_folder, constants.Constants.ODE_PDF),
            )
        if self._network_logger_path is not None:
            self._make_plot(
                dfs={constants.Constants.SIM: self._network_logger},
                save_path=os.path.join(
                    self._save_folder, constants.Constants.NETWORK_PDF
                ),
            )
        if self._ode_logger_path is not None and self._network_logger_path is not None:
            self._make_plot(
                dfs={
                    constants.Constants.ODE: self._ode_logger,
                    constants.Constants.SIM: self._network_logger,
                },
                save_path=os.path.join(
                    self._save_folder, constants.Constants.OVERLAY_PDF
                ),
            )

    def _make_overlay_plots(self):
        pass

    def _collate_logs(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Group columns from dataframe together according to identifiers.

        Args:
            df: dataframe whose columns are to be grouped according to identifiers.

        Returns:
            key_logs: dictionary of groups of df column tags
        """

        key_groups = {}

        identifiers = list(self.ERROR_IDENTIFIERS.keys())
        if self._log_overlaps:
            identifiers.extend(list(self.OVERLAP_IDENTIFIERS.keys()))

        for identifier in identifiers:
            tags = [col for col in df.columns if col.startswith(identifier)]
            key_groups[identifier] = tags

        return key_groups

    @staticmethod
    def get_figure_skeleton(
        height: Union[int, float],
        width: Union[int, float],
        num_columns: int,
        num_rows: int,
    ) -> Tuple:

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

    def _make_plot(self, dfs: Dict[str, pd.DataFrame], save_path: str) -> None:

        # can use arbitrary dataframe since columns will be the same.
        key_groups = self._collate_logs(df=list(dfs.values())[0])

        graph_layout = (3, 3)
        num_graphs = len(key_groups)
        num_rows = graph_layout[0]
        num_columns = graph_layout[1]

        self.fig, self.spec = self.get_figure_skeleton(
            height=4, width=5, num_columns=num_columns, num_rows=num_rows
        )

        for row in range(num_rows):
            for col in range(num_columns):

                graph_index = (row) * num_columns + col

                if graph_index < num_graphs:

                    print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))
                    group_name = list(key_groups.keys())[graph_index]
                    keys = list(key_groups.values())[graph_index]
                    self._plot_scalar(
                        row=row, col=col, dfs=dfs, group_name=group_name, keys=keys
                    )

        self.fig.savefig(save_path, dpi=100)

    def _plot_scalar(
        self,
        row: int,
        col: int,
        dfs: Dict[str, pd.DataFrame],
        group_name: str,
        keys: List[str],
    ):

        fig_sub = self.fig.add_subplot(self.spec[row, col])

        # base range of colors
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        # additional set of colors
        colors.extend(list(mcolors.CSS4_COLORS.values()))
        # create cyclical iterator to ensure sufficient colors
        color_cycle = cycle(colors)

        color_dict = {}
        for key in keys:
            color_dict[key] = next(color_cycle)

        for df_tag, df in dfs.items():
            if df_tag == constants.Constants.ODE:
                linestyle = constants.Constants.SOLID
            elif df_tag == constants.Constants.SIM:
                linestyle = constants.Constants.DASHED
            for key in keys:
                data_indexing = "".join(
                    [s for s in key.split(group_name)[1].split("_") if s.isdigit()]
                )
                data = df[key].dropna()
                scaling = self._num_steps / len(data)
                fig_sub.plot(
                    scaling * np.arange(len(data)),
                    data,
                    label=f"{self.IDENTIFIERS[group_name]} {data_indexing} {df_tag}",
                    color=color_dict[key],
                    linestyle=linestyle,
                )

        # unique_data_tags = np.unique([log.split(" ")[0] for log in logs.keys()])
        # colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # color_dict = {
        #     unique_tag: color for unique_tag, color in zip(unique_data_tags, colors)
        # }

        # for data_index, data in logs.items():

        #     split_data_index = data_index.split(" ")
        #     data_color = color_dict[split_data_index[0]]
        #     if split_data_index[1] == "Sim":
        #         linestyle = "dashed"
        #     else:
        #         linestyle = "solid"

        #     scale = self._num_steps / len(data)

        #     fig_sub.plot(
        #         scale * np.arange(len(data)),
        #         data,
        #         label=f"{log_key}_{data_index}",
        #         color=data_color,
        #         linestyle=linestyle,
        #     )

        # labelling
        fig_sub.set_xlabel(constants.Constants.STEP)
        fig_sub.set_ylabel(self.IDENTIFIERS[group_name])
        fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which="major", linestyle="-", linewidth="0.5", color="red", alpha=0.2
        )
        fig_sub.grid(
            which="minor", linestyle=":", linewidth="0.5", color="black", alpha=0.4
        )
