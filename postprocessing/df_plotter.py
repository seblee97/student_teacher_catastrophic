from utils import Parameters, get_figure_skeleton, smooth_data
from post_processing.plot_config import PlotConfigGenerator
from constants import Constants

import os
import pandas as pd
import itertools
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except OSError:
    pass

from typing import List


class DataFramePlotter:

    def __init__(self, config: Parameters):

        self._extract_parameters(config)
        self._get_data()

        self._plot_config = \
            PlotConfigGenerator.generate_plotting_config(
                config, gradient_overlaps=self._gradient_overlaps,
                overlap_differences=self._overlap_differences
                )

        self._setup_plotting()

    def _extract_parameters(self, config: Parameters) -> None:
        self._figure_name = config.get("df_figure_name")
        self._repeats = config.get("repeats")
        self._save_path = config.get("save_path")
        self.experiment_name = config.get("experiment_name")
        self._figure_save_path = os.path.join(self._save_path, "figures")

        self.crop_x = config.get(["post_processing", "crop_x"])
        self.combine_plots = \
            config.get(["post_processing", "combine_plots"])
        self.show_legends = \
            config.get(["post_processing", "show_legends"])
        self.plot_linewidth = \
            config.get(["post_processing", "plot_thickness"])

        self.width = config.get(["post_processing", "plot_width"])
        self.height = config.get(["post_processing", "plot_height"])

    def _get_data(self):
        if self._repeats:
            data_logger_paths = [
                os.path.join(self._save_path, f, "data_logger.csv")
                for f in os.listdir(self._save_path)
                if f != 'figures' and f != '.DS_Store'
                ]
            self._data = [
                pd.read_csv(data_logger_path)
                for data_logger_path in data_logger_paths
            ]
            columns = self._data[0].columns
        else:
            self._data = pd.read_csv(
                os.path.join(self._save_path, "data_logger.csv")
                )
            columns = self._data.columns

        self._gradient_overlaps = any('grad' in key for key in columns)
        self._overlap_differences = any('difference' in key for key in columns)

    def _setup_plotting(self) -> None:

        self.plot_keys = list(self._plot_config.keys())

        self.number_of_graphs = len(self.plot_keys)

        graph_layout = Constants.GRAPH_LAYOUTS[self.number_of_graphs]
        self.num_rows = graph_layout[0]
        self.num_columns = graph_layout[1]

        if self.combine_plots:
            self.fig, self.spec = get_figure_skeleton(
                height=self.height, width=self.width, num_columns=self.num_columns,
                num_rows=self.num_rows
                )

    def make_summary_plot(self):

        # make sub folder in results folder for figures
        os.makedirs(self._figure_save_path, exist_ok=True)

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
                "{}/{}.pdf".format(self._figure_save_path, self._figure_name),
                dpi=50
                )
            plt.close()

    def _get_scalar_data(self, attribute_keys: List[str]):
        if self._repeats:
            plot_data = [
                [
                    data[attribute_key].dropna().tolist()
                    for attribute_key in attribute_keys
                    ]
                for data in self._data
                ]
        else:
            plot_data = [
                self._data[attribute_key].dropna().tolist()
                for attribute_key in attribute_keys
                ]

        return plot_data

    def _get_image_data(self, attribute_keys: List):
        if self._repeats:
            plot_data = [
                {
                    index: data[attribute_keys[index]].dropna().tolist()
                    for index in attribute_keys
                }
                for data in self._data
                ]
        else:
            plot_data = {
                    index: self._data[attribute_keys[index]].dropna().tolist()
                    for index in attribute_keys
                }

        return plot_data

    def _create_subplot(self, row: int, col: int, graph_index: int):

        attribute_title = self.plot_keys[graph_index]
        attribute_config = \
            self._plot_config[attribute_title]
        attribute_keys = attribute_config['keys']
        attribute_plot_type = attribute_config['plot_type']
        attribute_labels = attribute_config['labels']
        plot_colours = attribute_config.get("colours")
        smoothing = attribute_config.get('smoothing')
        transform_data = attribute_config.get('transform_data')

        scale_axes = len(self._data)

        if attribute_plot_type == 'scalar':
            plot_data = self._get_scalar_data(attribute_keys)
            if smoothing is not None:
                plot_data = smooth_data(plot_data, smoothing)
            if transform_data is not None:
                plot_data = transform_data(plot_data)
            self.add_scalar_plot(
                plot_data=plot_data, row_index=row, column_index=col,
                title=attribute_title, labels=attribute_labels,
                scale_axes=scale_axes, colours=plot_colours
                )

        elif attribute_plot_type == 'image':
            if not self._repeats:
                plot_data = self._get_image_data(attribute_keys)
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

        colour_cycle = itertools.cycle(colours)

        if self.combine_plots:
            fig_sub = self.fig.add_subplot(self.spec[row_index, column_index])
        else:
            fig, fig_sub = plt.subplots()

        if self._repeats:
            plot_data, deviations = self._average_datasets(plot_data)

        for d, dataset in enumerate(plot_data):
            if len(dataset):
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

                current_cycle_color = next(colour_cycle)

                fig_sub.plot(
                    x_data[x_data_indices[0]:x_data_indices[1]],
                    dataset[x_data_indices[0]:x_data_indices[1]],
                    label=labels[d], linewidth=self.plot_linewidth,
                    color=current_cycle_color
                    )
                if self._repeats:
                    plus_deviation = \
                        (dataset + deviations[d])[
                            x_data_indices[0]:x_data_indices[1]
                            ]
                    minus_deviation = \
                        (dataset - deviations[d])[
                            x_data_indices[0]:x_data_indices[1]]

                    # only sample every 100th point for fill
                    fig_sub.fill_between(
                        x_data[x_data_indices[0]:x_data_indices[1]][::500],
                        minus_deviation[::500],
                        plus_deviation[::500],
                        color=current_cycle_color,
                        alpha=0.3
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
                "{}/{}_{}.pdf".format(
                    self._figure_save_path, title,
                    self._figure_name
                    ), dpi=50
            )
            plt.close()

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
            plt.close()