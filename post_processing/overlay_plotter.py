import numpy as np
import pandas as pd
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except:
    pass
from typing import Tuple, Union, Dict


class OverlayPlotter:

    def __init__(self, ode_csv_path: str, sim_csv_path: str):
        self._ode_df, self._sim_df = self._load_logs(ode_csv_path, sim_csv_path)
        self._collated_dict = self._organise_logs(self._ode_df, self._sim_df)

        # save collated dict with organised logs to csv
        path_head = os.path.split(ode_csv_path)[0]
        for key, dictionary in self._collated_dict.items():
            df_path = os.path.join(path_head, f"organised_logs_{key}.csv")
            pd.DataFrame(dictionary).to_csv(df_path)

        self._num_steps = len(self._sim_df)

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
        log_key = list(self._collated_dict.keys())[graph_index]

        logs = self._collated_dict[log_key]

        fig_sub = self.fig.add_subplot(self.spec[row, col])

        unique_data_tags = np.unique([log.split(" ")[0] for log in logs.keys()])
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        color_dict = {unique_tag: color for unique_tag, color in zip(unique_data_tags, colors)}

        for data_index, data in logs.items():

            split_data_index = data_index.split(" ")
            data_color = color_dict[split_data_index[0]]
            if split_data_index[1] == "Sim":
                linestyle = "dashed"
            else:
                linestyle = "solid"

            scale = self._num_steps / len(data)

            fig_sub.plot(
                scale * np.arange(len(data)),
                data,
                label=f"{log_key}_{data_index}",
                color=data_color,
                linestyle=linestyle)

        # labelling
        fig_sub.set_xlabel("Step")
        fig_sub.set_ylabel(log_key)
        fig_sub.legend()

        # grids
        fig_sub.minorticks_on()
        fig_sub.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2)
        fig_sub.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.4)

    def _load_logs(self, ode_csv_path: str, sim_csv_path: str) -> Tuple[pd.DataFrame]:
        ode_df = pd.read_csv(ode_csv_path)
        sim_df = pd.read_csv(sim_csv_path)
        return ode_df, sim_df

    def _organise_logs(self, ode_df: pd.DataFrame, sim_df: pd.DataFrame) -> Dict:
        # ode data log
        q_logs_ode = {
            f"{key.split('_')[1]} ODE": ode_df[key]
            for key in ode_df.columns
            if key.startswith("Q") and "diff" not in key
        }
        u_logs_ode = {
            f"{key.split('_')[1]} ODE": ode_df[key]
            for key in ode_df.columns
            if key.startswith("U") and "diff" not in key
        }
        r_logs_ode = {
            f"{key.split('_')[1]} ODE": ode_df[key]
            for key in ode_df.columns
            if key.startswith("R") and "diff" not in key
        }
        h1_logs_ode = {
            f"{key.split('_')[1]} ODE": ode_df[key]
            for key in ode_df.columns
            if key.startswith("h1") and "diff" not in key
        }
        h2_logs_ode = {
            f"{key.split('_')[1]} ODE": ode_df[key]
            for key in ode_df.columns
            if key.startswith("h2") and "diff" not in key
        }
        linear_error_logs_ode = {
            f"{i} ODE": ode_df[key] for i, key in enumerate([
                filter_k for filter_k in ode_df.columns
                if filter_k.startswith("error_linear") and "diff" not in filter_k
            ])
        }
        log_error_logs_ode = {
            f"{i} ODE": ode_df[key] for i, key in enumerate([
                filter_k for filter_k in ode_df.columns
                if filter_k.startswith("error_log") and "diff" not in filter_k
            ])
        }

        # simulation data log
        q_logs_sim = {
            "{} Sim".format("".join(key.split('/')[1].split('_')[1:])): sim_df[key].dropna()
            for key in sim_df.columns
            if "layer_0_student_self_overlap" in key
        }
        u_logs_sim = {
            "{} Sim".format("".join(key.split('/')[2].split('_')[1:])): sim_df[key].dropna()
            for key in sim_df.columns
            if key.startswith("layer_0_student_teacher_overlaps/1/values")
        }
        r_logs_sim = {
            "{} Sim".format("".join(key.split('/')[2].split('_')[1:])): sim_df[key].dropna()
            for key in sim_df.columns
            if key.startswith("layer_0_student_teacher_overlaps/0/values")
        }
        h1_logs_sim = {
            f"{key.split('_')[-1]} Sim": sim_df[key].dropna()
            for key in sim_df.columns
            if "student_head_0" in key and "_weight_" in key
        }
        h2_logs_sim = {
            f"{key.split('_')[-1]} Sim": sim_df[key].dropna()
            for key in sim_df.columns
            if "student_head_1" in key and "_weight_" in key
        }
        linear_error_logs_sim = {
            f"{key.split('_')[1]} Sim": sim_df[key].dropna()
            for key in sim_df.columns
            if "generalisation_error/linear" in key and "teacher" in key
        }
        log_error_logs_sim = {
            f"{key.split('_')[1]} Sim": sim_df[key].dropna()
            for key in sim_df.columns
            if "generalisation_error/log" in key and "teacher" in key
        }

        collated_dict = {
            **{
                "Q": {
                    **q_logs_ode,
                    **q_logs_sim
                }
            },
            **{
                "U": {
                    **u_logs_ode,
                    **u_logs_sim
                }
            },
            **{
                "R": {
                    **r_logs_ode,
                    **r_logs_sim
                }
            },
            **{
                "h1": {
                    **h1_logs_ode,
                    **h1_logs_sim
                }
            },
            **{
                "h2": {
                    **h2_logs_ode,
                    **h2_logs_sim
                }
            },
            **{
                "Error_Linear": {
                    **linear_error_logs_ode,
                    **linear_error_logs_sim
                }
            },
            **{
                "Error_Log": {
                    **log_error_logs_ode,
                    **log_error_logs_sim
                }
            },
        }

        return collated_dict

    def make_plot(self, save_path: str) -> None:

        graph_layout = (5, 2)
        num_graphs = len(self._collated_dict)
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

        self.fig.savefig(save_path, dpi=100)
