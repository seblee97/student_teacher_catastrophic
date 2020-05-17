from utils import Parameters
from constants import Constants

import os
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import copy

from typing import List


class StudentTeacherPostprocessor:

    def __init__(self, save_path: str, plot_config_path: str):

        self._save_path = save_path
        self._plot_config_path = plot_config_path
        self._config = self._retrieve_config()

        self.experiment_name = self._config.get("experiment_name")

    def _setup_plotting(self) -> None:

        with open(self._plot_config_path) as json_file:
            self._plot_config = json.load(json_file)

        self.plot_keys = list(self._plot_config["data"].keys())

        self.number_of_graphs = len(self.plot_keys)

        self.num_rows = Constants.GRAPH_LAYOUTS[self.number_of_graphs][0]
        self.num_columns = Constants.GRAPH_LAYOUTS[self.number_of_graphs][1]

        width = self._plot_config['config']['width']
        height = self._plot_config['config']['height']

        heights = [height for _ in range(self.num_rows)]
        widths = [width for _ in range(self.num_columns)]

        self.fig = plt.figure(
            constrained_layout=False,
            figsize=(self.num_columns * width, self.num_rows * height)
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
            return

        ordered_df_paths = sorted(
            all_df_paths,
            key=lambda x: float(x.split("iter_")[-1].strip(".csv"))
            )
        all_dfs = [pd.read_csv(df_path) for df_path in ordered_df_paths]
        merged_df = pd.concat(all_dfs)

        key_set = set()
        for df in all_dfs:
            key_set.update(df.keys())

        assert set(merged_df.keys()) == key_set, \
            "Merged df does not have correct keys"

        merged_df.to_csv(os.path.join(self._save_path, "data_logger.csv"))

        # remove individual dataframes
        for df in all_df_paths:
            os.remove(df)

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

        self.fig.suptitle("Summary Plot: {}".format(self.experiment_name))

        self.fig.savefig(
            "{}/summary_plot.pdf".format(self._figure_save_path), dpi=500
            )
        plt.close()

    def _create_subplot(self, row: int, col: int, graph_index: int):

        attribute_title = self.plot_keys[graph_index]
        attribute_config = \
            self._plot_config["data"][attribute_title]
        attribute_plot_type = attribute_config['plot_type']
        attribute_key_format = attribute_config['key_format']
        attribute_scale_axes = attribute_config.get("scale_axes")

        if attribute_key_format == 'uniform':

            attribute_keys = attribute_config['keys']
            attribute_labels = attribute_config['labels']

        elif attribute_key_format == 'recursive':

            base_attribute_key = attribute_config['keys']
            fill_ranges = [
                list(range(r)) for r in attribute_config['key_format_ranges']
                ]
            fill_combos = list(itertools.product(*fill_ranges))

            if attribute_plot_type == "scalar":
                attribute_keys = []
                attribute_labels = []
                for fill_combo in fill_combos:
                    attribute_key = copy.deepcopy(base_attribute_key)
                    for i, fill in enumerate(fill_combo):
                        attribute_key = \
                            attribute_key.replace('%', str(fill), 1)
                    attribute_keys.append(attribute_key)
                    attribute_labels.append(tuple(fill_combo))

            elif attribute_plot_type == 'image':

                attribute_keys = {}
                for fill_combo in fill_combos:
                    attribute_key = copy.deepcopy(base_attribute_key)
                    for i, fill in enumerate(fill_combo):
                        attribute_key = \
                            attribute_key.replace('%', str(fill), 1)
                    attribute_keys[tuple(fill_combo)] = attribute_key

        else:
            raise ValueError(
                "Key format {} not recognized".format(attribute_key_format)
            )

        if attribute_plot_type == 'scalar':
            plot_data = [
                self._data[attribute_key].dropna().tolist()
                for attribute_key in attribute_keys
                ]
            self.add_subplot(
                plot_data=plot_data, row_index=row, column_index=col,
                title=attribute_title, labels=attribute_labels,
                scale_axes=attribute_scale_axes
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

    def add_subplot(
        self,
        plot_data,
        row_index: int,
        column_index: int,
        title: str,
        labels: List,
        scale_axes: int
    ) -> None:

        if len(labels) > 10:
            linewidth = 0.05
        else:
            linewidth = 1

        fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])

        for d, dataset in enumerate(plot_data):
            # scale axes
            if scale_axes:
                scaling = scale_axes / len(dataset)
                x_data = [i * scaling for i in range(len(dataset))]
            else:
                x_data = range(len(dataset))

            fig_sub.plot(x_data, dataset, label=labels[d], linewidth=linewidth)

        # labelling
        fig_sub.set_xlabel("Step")
        fig_sub.set_ylabel(title)
        if len(labels) < 9:
            fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(
            which='major', linestyle='-', linewidth='0.5',
            color='red', alpha=0.5
            )
        fig_sub.grid(
            which='minor', linestyle=':', linewidth='0.5',
            color='black', alpha=0.5
            )

    def add_image(
        self,
        plot_data,
        matrix_dimensions,
        row_index: int,
        column_index: int,
        title: str
    ) -> None:

        fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])

        matrix = np.zeros(matrix_dimensions)

        for i in range(matrix_dimensions[0]):
            for j in range(matrix_dimensions[1]):
                matrix[i][j] = plot_data[(i, j)][-1]

        im = fig_sub.imshow(matrix, vmin=0, vmax=1)

        # colorbar
        divider = make_axes_locatable(fig_sub)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        self.fig.colorbar(im, cax=cax, orientation='vertical')

        # title and ticks
        fig_sub.set_ylabel(title)
        fig_sub.set_xticks([])
        fig_sub.set_yticks([])
