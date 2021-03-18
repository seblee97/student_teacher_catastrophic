import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

from typing import Dict, List, Optional, Tuple, Union


class TeacherTeacherPlotter:

    ODE_DF_NAME = "ode_logs.csv"
    SIM_DF_NAME = "data_logger.csv"

    def __init__(self, folder: str, overlaps: np.ndarray, ode: bool, sim: bool):
        if ode:
            self._ode_dfs = self._get_dfs(folder, self.ODE_DF_NAME)
        if sim:
            self._sim_dfs = self._get_dfs(folder, self.SIM_DF_NAME)
        self._overlaps = overlaps

    @staticmethod
    def _get_dfs(folder: str, df_name: str) -> Dict[str, pd.DataFrame]:
        dfs = {}
        indices = os.listdir(folder)
        for index in indices:
            if index != ".DS_Store" and os.path.isdir(os.path.join(folder, index)):
                try:
                    dfs[int(index)] = pd.read_csv(f"{folder}/{index}/{df_name}")
                except FileNotFoundError:
                    print(f"df with name {df_name} not found at index {index}")
        return dfs

    def plot_vs_step(self,
                     data_origin: str,
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

        if data_origin == "ode":
            try:
                dfs = self._ode_dfs
            except AttributeError:
                raise ValueError("No ode dfs, cannot make plots with data origin ode")
        elif data_origin == "sim":
            try:
                dfs = self._sim_dfs
            except AttributeError:
                raise ValueError("No sim dfs, cannot make plots with data origin sim")
        else:
            raise ValueError("data origin {data_origin} invalid. Use 'ode' or 'sim'")

        fig = plt.figure(figsize=figsize)
        for attribute, label, color_map in zip(attributes, labels, color_maps):
            for i in sorted(list(dfs.keys())):
                dfs[i][attribute].plot(
                    label=f"V: {round(self._overlaps[i], 4)}; {label}",
                    color=color_map(i / len(self._overlaps)),
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
                        data_origin: str,
                        attribute: str,
                        steps: Union[List[int], int],
                        color: Union[str, matplotlib.colors.ListedColormap],
                        alpha: float = 1,
                        linewidth: Union[float, int] = 5,
                        show_axis_labels: bool = True,
                        show_ticks: bool = True,
                        major_gridlines: bool = True,
                        minor_gridlines: bool = True,
                        xlims: Tuple = (0, 1),
                        ylims: Optional[Tuple] = None,
                        save_name: Optional[str] = None):

        if data_origin == "ode":
            try:
                dfs = self._ode_dfs
            except AttributeError:
                raise ValueError("No ode dfs, cannot make plots with data origin ode")
        elif data_origin == "sim":
            try:
                dfs = self._sim_dfs
            except AttributeError:
                raise ValueError("No sim dfs, cannot make plots with data origin sim")
        else:
            raise ValueError("data origin {data_origin} invalid. Use 'ode' or 'sim'")

        if isinstance(steps, int):
            attribute_values_at_step = [
                dfs[i][attribute][steps] for i in range(len(self._overlaps))
            ]
        elif isinstance(steps, list):
            attribute_values_at_step = [
                dfs[i][attribute][steps[0]] - dfs[i][attribute][steps[1]]
                for i in range(len(self._overlaps))
            ]
        fig = plt.figure()
        if isinstance(color, str):
            plt.plot(
                self._overlaps,
                attribute_values_at_step,
                linewidth=linewidth,
                alpha=alpha,
                color=color)
        elif isinstance(color, matplotlib.colors.ListedColormap):
            f = interp1d(self._overlaps, attribute_values_at_step)
            x = np.linspace(0, 1, 10000)
            y = f(x)
            colors = [color(i) for i in x]
            for overlap, attribute_value, color in zip(x, y, colors):
                plt.scatter(overlap, attribute_value, color=color)
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
