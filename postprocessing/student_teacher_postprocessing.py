from utils import Parameters
from constants import Constants
from postprocessing.plot_config import PlotConfigGenerator

import os
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import List


class StudentTeacherPostprocessor:

    def __init__(self, save_path: str, plot_config_path: str):

        self._save_path = save_path
        self._plot_config_path = plot_config_path
        self._config = self._retrieve_config()

        self.experiment_name = self._config.get("experiment_name")

    def _setup_plotting(self) -> None:

        self.crop_x = self._config.get(["post_processing", "crop_x"])
        self.combine_plots = \
            self._config.get(["post_processing", "combine_plots"])
        self.show_legends = \
            self._config.get(["post_processing", "show_legends"])
        self.plot_linewidth = \
            self._config.get(["post_processing", "plot_thickness"])

        self._plot_config = \
            PlotConfigGenerator.generate_plotting_config(self._config)

        self.plot_keys = list(self._plot_config.keys())

        self.number_of_graphs = len(self.plot_keys)

        self.num_rows = Constants.GRAPH_LAYOUTS[self.number_of_graphs][0]
        self.num_columns = Constants.GRAPH_LAYOUTS[self.number_of_graphs][1]

        self.width = self._config.get(["post_processing", "plot_width"])
        self.height = self._config.get(["post_processing", "plot_height"])

        heights = [self.height for _ in range(self.num_rows)]
        widths = [self.width for _ in range(self.num_columns)]

        if self.combine_plots:

            self.fig = plt.figure(
                constrained_layout=False,
                figsize=(
                    self.num_columns * self.width,
                    self.num_rows * self.height
                    )
                )

            self.spec = gridspec.GridSpec(
                nrows=self.num_rows, ncols=self.num_columns,
                width_ratios=widths, height_ratios=heights
                )

    def _retrieve_config(self):
        # read parameters from config in save path
        config_path = os.path.join(self._save_path, "config.yaml")
        with open(config_path, 'r') as yaml_file:
            params = yaml.load(yaml_file, yaml.SafeLoader)

        # create object in which to store experiment parameters and
        # validate config file
        config_parameters = Parameters(params)

        return config_parameters

    def post_process(self) -> None:

        self._consolidate_dfs()

        self._data = pd.read_csv(
            os.path.join(self._save_path, "data_logger.csv")
            )

        self._setup_plotting()
        self._make_summary_plot()

    def _consolidate_dfs(self):
        print("Consolidating/Merging all dataframes..")
        all_df_paths = [
            os.path.join(self._save_path, f)
            for f in os.listdir(self._save_path) if f.endswith('.csv')
            ]

        if any("data_logger.csv" in path for path in all_df_paths):
            print(
                "'data_logger.csv' file already in save path "
                "specified. Consolidation already complete."
                )
            if len(all_df_paths) > 1:
                print("Note, other csv files are also still in save path.")

        else:
            ordered_df_paths = sorted(
                all_df_paths,
                key=lambda x: float(x.split("iter_")[-1].strip(".csv"))
                )

            print("Loading all dataframes..")
            all_dfs = [pd.read_csv(df_path) for df_path in ordered_df_paths]
            print("Dataframes loaded. Merging..")
            merged_df = pd.concat(all_dfs)

            key_set = set()
            for df in all_dfs:
                key_set.update(df.keys())

            assert set(merged_df.keys()) == key_set, \
                "Merged df does not have correct keys"

            print("Dataframes merged. Saving...")
            merged_df.to_csv(os.path.join(self._save_path, "data_logger.csv"))

            print("Saved. Removing individual dataframes...")
            # remove individual dataframes
            for df in all_df_paths:
                os.remove(df)
            print("Consolidation complete.")

    def _make_summary_plot(self):

        # make sub folder in results folder for figures
        os.makedirs("{}/figures/".format(self._save_path), exist_ok=True)
        self._figure_save_path = os.path.join(self._save_path, "figures")

        for row in range(self.num_rows):
            for col in range(self.num_columns):

                graph_index = (row) * self.num_columns + col

                if graph_index < self.number_of_graphs:

                    print(
                        "Plotting graph {}/{}".format(
                            graph_index + 1, self.number_of_graphs
                            )
                        )

                    self._create_subplot(
                        row=row,
                        col=col,
                        graph_index=graph_index
                        )

        if self.combine_plots:
            self.fig.suptitle("Summary Plot: {}".format(self.experiment_name))

            self.fig.savefig(
                "{}/summary_plot.pdf".format(self._figure_save_path), dpi=500
                )
            plt.close()

    def _create_subplot(self, row: int, col: int, graph_index: int):

        attribute_title = self.plot_keys[graph_index]
        attribute_config = \
            self._plot_config[attribute_title]
        attribute_keys = attribute_config['keys']
        attribute_plot_type = attribute_config['plot_type']
        attribute_labels = attribute_config['labels']
        plot_colours = attribute_config.get("colours")

        scale_axes = len(self._data)

        if attribute_plot_type == 'scalar':
            plot_data = [
                self._data[attribute_key].dropna().tolist()
                for attribute_key in attribute_keys
                ]
            self.add_scalar_plot(
                plot_data=plot_data, row_index=row, column_index=col,
                title=attribute_title, labels=attribute_labels,
                scale_axes=scale_axes, colours=plot_colours
                )

        elif attribute_plot_type == 'image':
            plot_data = {
                index: self._data[attribute_keys[index]].dropna().tolist()
                for index in attribute_keys
                }
            self.add_image(
                plot_data=plot_data,
                matrix_dimensions=tuple(
                    attribute_config['key_format_ranges']
                    ),
                row_index=row, column_index=col,
                title=attribute_title
            )

        else:
            raise ValueError(
                "Plot type {} not recognized".format(attribute_plot_type)
            )

    def add_scalar_plot(
        self,
        plot_data,
        row_index: int,
        column_index: int,
        title: str,
        labels: List,
        scale_axes: int,
        colours: List[str]
    ) -> None:

        # if len(labels) > 10:
        #     linewidth = 0.05
        # else:
        #     linewidth = 1

        if self.combine_plots:
            fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])
        else:
            fig, fig_sub = plt.subplots()

        for d, dataset in enumerate(plot_data):
            # scale axes
            if scale_axes:
                scaling = scale_axes / len(dataset)
                x_data = [i * scaling for i in range(len(dataset))]
            else:
                x_data = range(len(dataset))

            if self.crop_x:
                uncropped_x_data_len = len(x_data)
                x_data_indices = [
                    round(self.crop_x[0] * uncropped_x_data_len),
                    round(self.crop_x[1] * uncropped_x_data_len)
                    ]
            else:
                x_data_indices = [0, len(x_data)]

            fig_sub.plot(
                x_data[x_data_indices[0]:x_data_indices[1]],
                dataset[x_data_indices[0]:x_data_indices[1]],
                label=labels[d], linewidth=self.plot_linewidth,
                color=colours[d]
                )

        # labelling
        fig_sub.set_xlabel("Step")
        fig_sub.set_ylabel(title)
        if len(labels) < 9 and self.show_legends:
            fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which='major', linestyle='-', linewidth='0.5',
            color='red', alpha=0.2
            )
        fig_sub.grid(
            which='minor', linestyle=':', linewidth='0.5',
            color='black', alpha=0.4
            )

        if not self.combine_plots:
            fig.savefig(
                "{}/{}.pdf".format(self._figure_save_path, title), dpi=500
            )

    def add_image(
        self,
        plot_data,
        matrix_dimensions,
        row_index: int,
        column_index: int,
        title: str
    ) -> None:

        if self.combine_plots:
            fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])
        else:
            fig, fig_sub = plt.subplots()

        matrix = np.zeros(matrix_dimensions)

        for i in range(matrix_dimensions[0]):
            for j in range(matrix_dimensions[1]):
                matrix[i][j] = plot_data[(i, j)][-1]

        im = fig_sub.imshow(matrix, vmin=0, vmax=1)

        # colorbar
        divider = make_axes_locatable(fig_sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        if self.combine_plots:
            self.fig.colorbar(im, cax=cax, orientation='vertical')
        else:
            fig.colorbar(im, cax=cax, orientation='vertical')

        # title and ticks
        fig_sub.set_ylabel(title)
        fig_sub.set_xticks([])
        fig_sub.set_yticks([])

        if not self.combine_plots:
            fig.savefig(
                "{}/{}.pdf".format(self._figure_save_path, title), dpi=500
            )
