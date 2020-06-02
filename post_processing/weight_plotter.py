from utils import Parameters, get_figure_skeleton, close_sqrt_factors
from constants import Constants

import os
import torch
import numpy as np

try:
    import matplotlib.pyplot as plt
except OSError:
    pass

from typing import List


class WeightPlotter:

    def __init__(self, config: Parameters):

        self._extract_parameters(config)

        self.loaded_weights = self._get_saved_weights()

    def _extract_parameters(self, config: Parameters):
        self._save_path = config.get("save_path")
        self._figure_save_path = os.path.join(self._save_path, "figures")

        self.hidden_layers = config.get(["model", "student_hidden_layers"])

        self.width = config.get(["post_processing", "plot_width"])
        self.height = config.get(["post_processing", "plot_height"])

    def _get_saved_weights(self) -> List:

        weights_path = os.path.join(self._save_path, "learner_weights")
        saved_weight_files = [f for f in os.listdir(weights_path) if "switch" in f]

        sorted_saved_weight_files = sorted(
            saved_weight_files,
            key=lambda x: int(x.split("switch_")[1].split("_step")[0])
            )

        full_sorted_saved_weight_files = [
            os.path.join(weights_path, f)
            for f in sorted_saved_weight_files
            ]

        loaded_weights = [
            torch.load(weights) for weights in full_sorted_saved_weight_files
            ]

        return loaded_weights

    def make_per_layer_weight_plot(self, plot_type: str) -> None:

        if plot_type == "weight_pdf":
            plot_fn = self._weight_pdf_plot
            plot_title = "Weight Distribution PDFs"

            def _extract_num_plots(layer_weights):
                return len(layer_weights)

        elif plot_type == "weight_diffs":
            plot_fn = self._weight_diff_plot
            plot_title = "Weight Differences"

            def _extract_num_plots(layer_weights):
                return len(layer_weights) - 1

        elif plot_type == "weight_diff_pdf":
            plot_fn = self._weight_diff_pdf_plot
            plot_title = "Weight Difference PDFs"

            def _extract_num_plots(layer_weights):
                return len(layer_weights) - 1

        else:
            raise ValueError("Plot type {} not recognised".format(plot_type))

        # make sub folder in results folder for figures
        os.makedirs(self._figure_save_path, exist_ok=True)

        for layer_index in range(len(self.hidden_layers)):

            layer_weights = [
                weights['layers.{}.weight'.format(layer_index)].numpy()
                for weights in self.loaded_weights
                ]

            num_plots = _extract_num_plots(layer_weights)

            graph_layout = Constants.GRAPH_LAYOUTS.get(num_plots)

            if graph_layout is None:
                graph_layout = close_sqrt_factors(num_plots)

            num_rows = graph_layout[0]
            num_columns = graph_layout[1]

            fig, spec = get_figure_skeleton(
                height=self.height, width=self.width, num_columns=num_columns,
                num_rows=num_rows
                )

            for row in range(num_rows):
                for col in range(num_columns):
                    graph_index = (row) * num_columns + col
                    if graph_index < num_plots:

                        plot_fn(
                            fig=fig, spec=spec, row=row, col=col,
                            layer_weights=layer_weights,
                            graph_index=graph_index
                            )

            fig.suptitle("Layer {} {}".format(layer_index, plot_title))

            fig.savefig(
                "{}/layer_{}_{}.pdf".format(
                    self._figure_save_path, layer_index,
                    plot_type
                    ),
                dpi=50
                )

    def _weight_pdf_plot(
        self,
        fig,
        spec,
        row: int,
        col: int,
        layer_weights: np.ndarray,
        graph_index: int
    ) -> None:

        iteration_layer_weights = layer_weights[graph_index]

        fig_sub = fig.add_subplot(spec[row, col])
        fig_sub.hist(
            np.concatenate(iteration_layer_weights),
            bins=30, alpha=1, label="All weights"
        )

        for c, component in enumerate(iteration_layer_weights):
            fig_sub.hist(
                component, bins=30, alpha=1,
                label="Component {}".format(c)
                )

        fig_sub.set_xlabel("Weight magnitude")
        fig_sub.set_ylabel("Count")

        fig_sub.legend()
        fig_sub.set_title("@ switch {}".format(graph_index))

    def _weight_diff_plot(
        self,
        fig,
        spec,
        row: int,
        col: int,
        layer_weights: np.ndarray,
        graph_index: int
    ) -> None:

        weight_differences = \
            layer_weights[graph_index + 1] - layer_weights[graph_index]

        fig_sub = fig.add_subplot(spec[row, col])

        for c, component in enumerate(weight_differences):
            fig_sub.bar(
                range(len(component)),
                component,
                alpha=0.5,
                label="Component {}".format(c)
                )

        fig_sub.set_ylabel("Weight change")
        fig_sub.set_xlabel("Weight index")

        fig_sub.legend()
        fig_sub.set_title("@ switch {}".format(graph_index + 1))

    def _weight_diff_pdf_plot(
        self,
        fig,
        spec,
        row: int,
        col: int,
        layer_weights: np.ndarray,
        graph_index: int
    ) -> None:

        weight_differences = \
            layer_weights[graph_index + 1] - layer_weights[graph_index]

        fig_sub = fig.add_subplot(spec[row, col])

        fig_sub.hist(
            np.concatenate(weight_differences),
            bins=30, alpha=1, label="All weights"
        )

        for c, component in enumerate(weight_differences):
            fig_sub.hist(
                component, bins=30, alpha=1,
                label="Component {}".format(c)
                )

        fig_sub.set_ylabel("Count")
        fig_sub.set_xlabel("Weight change")

        fig_sub.legend()
        fig_sub.set_title("@ switch {}".format(graph_index + 1))
