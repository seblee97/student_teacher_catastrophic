import os
from typing import Dict
from typing import Union

import pandas as pd

import constants
from plotters import base_plotter


class UnifiedPlotter(base_plotter.BasePlotter):
    """Class for plotting generalisation errors, overlaps etc.

    For case when logging is done in 'unified' fashion i.e. all into one dataframe.
    """

    def __init__(
        self,
        save_folder: str,
        num_steps: int,
        log_overlaps: bool,
        log_ode: bool,
        log_network: bool,
    ):
        """
        Class constructor.

        Args:
            save_folder: path to folder for saving plots.
            num_steps: total number of steps in the training run (used for scaling axes).
            log_overlaps: whether or not to plot overlaps (or just errors).
            log_ode: whether ot not to plot ode data.
            log_network: whether ot not to plot network data.
        """
        self._ode_logger_path = (
            os.path.join(save_folder, constants.Constants.ODE_CSV) if log_ode else None
        )
        self._network_logger_path = (
            os.path.join(save_folder, constants.Constants.NETWORK_CSV)
            if log_network
            else None
        )

        super().__init__(
            save_folder=save_folder, num_steps=num_steps, log_overlaps=log_overlaps
        )

    def _setup_data(self):
        """Setup data from relevant dataframes.

        Here, in the unified case, full dataset is loaded into memory.
        """
        if self._ode_logger_path is not None:
            self._ode_logger = pd.read_csv(self._ode_logger_path)
        if self._network_logger_path is not None:
            self._network_logger = pd.read_csv(self._network_logger_path)

    def make_plots(self) -> None:
        """Orchestration method for plotting ode logs, network logs, or both."""
        if self._ode_logger_path is not None:
            self._make_plot(
                data={constants.Constants.ODE: self._ode_logger},
                save_path=os.path.join(self._save_folder, constants.Constants.ODE_PDF),
            )
        if self._network_logger_path is not None:
            self._make_plot(
                data={constants.Constants.SIM: self._network_logger},
                save_path=os.path.join(
                    self._save_folder, constants.Constants.NETWORK_PDF
                ),
            )
        if self._ode_logger_path is not None and self._network_logger_path is not None:
            self._make_plot(
                data={
                    constants.Constants.ODE: self._ode_logger,
                    constants.Constants.SIM: self._network_logger,
                },
                save_path=os.path.join(
                    self._save_folder, constants.Constants.OVERLAY_PDF
                ),
            )

    def _make_plot(
        self,
        data: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]],
        save_path: str,
    ) -> None:
        """Make plots for a set of results (e.g. ode or network or both).

        Args:
            data: mapping from type of results (ode, network etc.)
            to dataframes with results.
            save_path: path to save the plot.
        """
        # can use arbitrary dataframe since columns will be the same.
        tag_groups = self._collate_tags(tags=list(list(data.values())[0].keys()))

        # e.g. [error, overlap, ...]
        group_names = list(tag_groups.keys())
        # e.g. [[error_1, error_2, ...], [overlap_1, overlap_2, ...], ...]
        group_key_names = list(tag_groups.values())  # e.g.

        num_graphs = len(tag_groups)
        num_rows = self.GRAPH_LAYOUT[0]
        num_columns = self.GRAPH_LAYOUT[1]

        fig, spec = self._get_figure_skeleton(
            height=4, width=5, num_columns=num_columns, num_rows=num_rows
        )

        for row in range(num_rows):
            for col in range(num_columns):

                graph_index = (row) * num_columns + col

                if graph_index < num_graphs:

                    print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))
                    group_name = group_names[graph_index]
                    keys = group_key_names[graph_index]

                    data_collection = {
                        data_type: {
                            key: data[data_type][key].dropna().to_numpy()
                            for key in keys
                        }
                        for data_type in data.keys()
                    }

                    fig = self._plot_scalar(
                        fig=fig,
                        spec=spec,
                        row=row,
                        col=col,
                        tag_group_name=group_name,
                        data_collection=data_collection,
                    )

        fig.savefig(save_path, dpi=100)
