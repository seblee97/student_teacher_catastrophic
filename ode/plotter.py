from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, logs: List[Dict], scale: Union[int, float]):
        self._logs = logs
        self._log_keys = list(self._logs.keys())
        self._scale = scale

    @staticmethod
    def get_figure_skeleton(height: Union[int, float], width: Union[int, float], num_columns: int,
                            num_rows: int) -> Tuple:

        fig = plt.figure(constrained_layout=False, figsize=(num_columns * width, num_rows * height))

        heights = [height for _ in range(num_rows)]
        widths = [width for _ in range(num_columns)]

        spec = gridspec.GridSpec(
            nrows=num_rows, ncols=num_columns, width_ratios=widths, height_ratios=heights)

        return fig, spec

    def _plot_scalar(self, row: int, col: int, graph_index: int):
        log_key = self._log_keys[graph_index]
        logs = self._logs[log_key]

        fig_sub = self.fig.add_subplot(self.spec[row, col])

        for data_index, data in logs.items():

            fig_sub.plot(self._scale * np.arange(len(data)), data, label=f"{log_key}_{data_index}")

        # labelling
        fig_sub.set_xlabel("Step")
        fig_sub.set_ylabel(log_key)
        fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2)
        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.4)

    def plot(self):

        # 1. Errors (Linear) 2. Errors (Log)
        # 3. Q, 4. R, 5. U 6. h1, 7. h2

        graph_layout = (5, 2)
        num_graphs = len(self._logs)
        num_rows = graph_layout[0]
        num_columns = graph_layout[1]

        self.fig, self.spec = self.get_figure_skeleton(
            height=4, width=5, num_columns=num_columns, num_rows=num_rows)

        for row in range(num_rows):
            for col in range(num_columns):

                graph_index = (row) * num_columns + col

                if graph_index < num_graphs:

                    print("Plotting graph {}/{}".format(graph_index + 1, num_graphs))

                    self._plot_scalar(row=row, col=col, graph_index=graph_index)

        return self.fig
