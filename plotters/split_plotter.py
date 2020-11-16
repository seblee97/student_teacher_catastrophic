import os
from typing import Dict
from typing import List

import numpy as np
import pandas as pd

import constants
from plotters import base_plotter


class SplitPlotter(base_plotter.BasePlotter):
    """Class for plotting generalisation errors, overlaps etc.

    For case when logging is done in 'split' fashion
    i.e. one dataframe per logging tag.
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
        self._ode_data_folder = (
            os.path.join(save_folder, constants.Constants.ODE) if log_ode else None
        )
        self._network_data_folder = (
            os.path.join(save_folder, constants.Constants.SIM) if log_network else None
        )

        super().__init__(
            save_folder=save_folder, num_steps=num_steps, log_overlaps=log_overlaps
        )

    def _setup_data(self):
        """Setup data from relevant dataframes.

        Here, in the split case, we only establish paths to
        datasets without loading into memory--this is done piecewise later.
        """
        if self._ode_data_folder is not None:
            self._ode_log_file_paths = {
                path.split(f"_{constants.Constants.ODE_CSV}")[0]: os.path.join(
                    self._ode_data_folder, path
                )
                for path in os.listdir(self._ode_data_folder)
            }
        if self._network_data_folder is not None:
            self._network_log_file_paths = {
                path.split(f"_{constants.Constants.NETWORK_CSV}")[0]: os.path.join(
                    self._network_data_folder, path
                )
                for path in os.listdir(self._network_data_folder)
            }

    def make_plots(self) -> None:
        """Orchestration method for plotting ode logs, network logs, or both."""
        if self._ode_data_folder is not None:
            self._make_plot(
                data={constants.Constants.ODE: list(self._ode_log_file_paths.keys())},
                save_path=os.path.join(self._save_folder, constants.Constants.ODE_PDF),
            )
        if self._network_data_folder is not None:
            self._make_plot(
                data={
                    constants.Constants.SIM: list(self._network_log_file_paths.keys())
                },
                save_path=os.path.join(
                    self._save_folder, constants.Constants.NETWORK_PDF
                ),
            )
        if self._ode_data_folder is not None and self._network_data_folder is not None:
            self._make_plot(
                data={
                    constants.Constants.ODE: list(self._ode_log_file_paths.keys()),
                    constants.Constants.SIM: list(self._network_log_file_paths.keys()),
                },
                save_path=os.path.join(
                    self._save_folder, constants.Constants.OVERLAY_PDF
                ),
            )

    def _make_plot(
        self,
        data: Dict[str, List[str]],
        save_path: str,
    ) -> None:
        """Make plots for a set of results (e.g. ode or network or both).

        Args:
            data: mapping from type of results (ode, network etc.)
            to list of paths pointing to relevant dataframes.
            save_path: path to save the plot.
        """
        # can use arbitrary set of tags since columns will be the same.
        tag_groups = self._collate_tags(tags=list(data.values())[0])

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
                        data_type: self._load_data_from_keys(
                            data_type=data_type, keys=keys
                        )
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

    def _load_data_from_keys(
        self, data_type: str, keys: List[str]
    ) -> Dict[str, np.ndarray]:
        if data_type == constants.Constants.ODE:
            file_path_map = self._ode_log_file_paths
        elif data_type == constants.Constants.SIM:
            file_path_map = self._network_log_file_paths
        data = {
            key: pd.read_csv(file_path_map[key]).to_numpy().flatten() for key in keys
        }
        return data
