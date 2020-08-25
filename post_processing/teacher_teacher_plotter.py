import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Tuple, Union


class TeacherTeacherPlotter:

    def __init__(self, folder: str, overlaps: np.ndarray):
        self._dfs = self._get_dfs(folder)
        self._indices = sorted(self._dfs.keys())
        self._overlaps = overlaps

    @staticmethod
    def _get_dfs(folder: str) -> Dict[str, pd.DataFrame]:
        dfs = {}
        indices = os.listdir(folder)
        for index in indices:
            try:
                dfs[int(index)] = pd.read_csv(f"{folder}/{index}/ode_logs.csv")
            except FileNotFoundError:
                print(f"index {index} now found")
        return dfs

    def plot_vs_step(self,
                     attributes: List[str],
                     labels: List[str],
                     color_maps: List,
                     figsize: Tuple = (25, 16),
                     alpha: float = 0.8,
                     linewidth: Union[float, int] = 5,
                     show_axis_labels: bool = True,
                     show_legend: bool = True,
                     show_ticks: bool = True,
                     major_gridlines: bool = True,
                     minor_gridlines: bool = True,
                     save_name: Optional[str] = None,
                     xlims: Optional[Tuple] = None,
                     ylims: Optional[Tuple] = None):
        fig = plt.figure(figsize=figsize)
        for attribute, label, color_map in zip(attributes, labels, color_maps):
            for i, index in enumerate(self._indices):
                self._dfs[index][attribute].plot(
                    label=f"V: {round(self._overlaps[index], 4)}; {label}",
                    color=color_map(i / len(self._indices)),
                    alpha=alpha,
                    linewidth=linewidth)
        if show_axis_labels:
            plt.xlabel("Time Step")
        if show_legend:
            plt.legend()
        if xlims is not None:
            plt.xlim(xlims)
        if ylims is not None:
            plt.ylim(ylims)
        if major_gridlines:
            plt.grid(b=True, which='major', color='r', linestyle='-', alpha=0.2)
        if minor_gridlines:
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='gray', linestyle='--', alpha=0.2)
        if not show_ticks:
            plt.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                left=False,
                labelleft=False)
        if save_name is not None:
            fig.savefig(save_name, dpi=100, bbox_inches='tight', pad_inches=0)

    def plot_vs_overlap(self,
                        attribute: str,
                        steps: Union[List[int], int],
                        color: str,
                        alpha: float = 1,
                        linewidth: Union[float, int] = 5,
                        show_axis_labels: bool = True,
                        show_ticks: bool = True,
                        major_gridlines: bool = True,
                        minor_gridlines: bool = True,
                        xlims: Tuple = (0, 1),
                        ylims: Optional[Tuple] = None,
                        save_name: Optional[str] = None):
        if isinstance(steps, int):
            attribute_values_at_step = [self._dfs[i][attribute][steps] for i in self._indices]
        elif isinstance(steps, list):
            attribute_values_at_step = [
                self._dfs[i][attribute][steps[0]] - self._dfs[i][attribute][steps[1]]
                for i in self._indices
            ]
        fig = plt.figure()
        plt.plot(
            self._overlaps, attribute_values_at_step, linewidth=linewidth, alpha=alpha, color=color)
        if show_axis_labels:
            plt.xlabel("Teacher-Teacher Overlap")
        if xlims is not None:
            plt.xlim((0, 1))
        if ylims is not None:
            plt.ylim(ylims)
        if major_gridlines:
            plt.grid(b=True, which='major', color='r', linestyle='-', alpha=0.2)
        if minor_gridlines:
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='gray', linestyle='--', alpha=0.2)
        if not show_ticks:
            plt.tick_params(
                axis='both',
                which='both',
                bottom=False,
                top=False,
                labelbottom=False,
                left=False,
                labelleft=False)
        if save_name is not None:
            fig.savefig(save_name, dpi=100, bbox_inches='tight', pad_inches=0)
