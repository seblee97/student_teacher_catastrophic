"""
Post-processing script for sets of experiments varying feature similarity. 

E.g. for results folder containing the following sub-folders: 

feature_*_interleave_period_**

where * represents the feature similarity and ** the interleaved period.

This script will split the plots into **, and plot trajectories for different * in each plot.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_folder", type=str, help="path to results folder to post-process."
)
parser.add_argument(
    "--varying_attribute",
    type=str,
    help="name of attribute on which to split runs (** in description above, e.g. interleave_period).",
)
parser.add_argument(
    "--attribute_values",
    type=str,
    help="list of values the attribute takes in results. In format '[x, y, z]'",
)


def plot_cross_section(dfs, switch_step: int, T: int, varying_attribute: str, save_path: str, save_name: str):
    sorted_paths = sorted(
        dfs.keys(),
        key=lambda x: float(x.split("feature_")[1].split(f"_{varying_attribute}")[0]),
    )
    e0s = []
    e1s = []
    for feature in sorted_paths:
        switch_error_0 = dfs[feature].generalisation_error_0[switch_step]
        switch_error_1 = dfs[feature].generalisation_error_1[switch_step]
        e0 = dfs[feature].generalisation_error_0[T] - switch_error_0 
        e1 = switch_error_1 - dfs[feature].generalisation_error_1[T]
        e0s.append(e0)
        e1s.append(e1)

    fig = plt.figure()
    plt.title("Forgetting Cross")
    plt.plot(range(len(e0s)), e0s)
    fig.show()
    fig.savefig(os.path.join(save_path, f"{save_name}_{T}_forgetting_cross.pdf"))

    fig = plt.figure()
    plt.title("Transfer Cross")
    plt.plot(range(len(e1s)), e1s)
    fig.show()
    fig.savefig(os.path.join(save_path, f"{save_name}_{T}_transfer_cross.pdf"))


def plot_error_trajectories(dfs, varying_attribute: str, save_path: str, save_name: str):
    color_map_1 = cm.get_cmap("viridis")
    sorted_paths = sorted(
        dfs.keys(),
        key=lambda x: float(x.split("feature_")[1].split(f"_{varying_attribute}")[0]),
    )
    fig = plt.figure()
    for i, feature in enumerate(sorted_paths):
        dfs[feature].log_generalisation_error_0.plot(
            color=color_map_1(i / len(sorted_paths)),
            label=feature.split("feature_")[1].split(f"_{varying_attribute}")[0],
        )
    plt.title("First Error")
    plt.legend()
    fig.savefig(os.path.join(save_path, f"{save_name}_first_error.pdf"))

    color_map_2 = cm.get_cmap("plasma")
    fig = plt.figure()
    for i, feature in enumerate(sorted_paths):
        dfs[feature].log_generalisation_error_1.plot(
            color=color_map_2(i / len(sorted_paths)),
            label=feature.split("feature_")[1].split(f"_{varying_attribute}")[0],
        )
    plt.title("Second Error")
    plt.legend()
    fig.savefig(os.path.join(save_path, f"{save_name}_second_error.pdf"))


def plot_overlap_trajectories(dfs, varying_attribute: str, save_path: str, save_name: str):
    color_map_1 = cm.get_cmap("viridis")
    sorted_paths = sorted(
        dfs.keys(),
        key=lambda x: float(x.split("feature_")[1].split(f"_{varying_attribute}")[0]),
    )

    # self overlap, Q
    fig = plt.figure()
    for i, feature in enumerate(sorted_paths):
        label_base = feature.split("feature_")[1].split(f"_{varying_attribute}")[0]
        dfs[feature].student_self_overlap_0_0.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_00",
            linestyle="dashed"
        )
        dfs[feature].student_self_overlap_0_1.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_01",
            linestyle="dotted"
        )
        dfs[feature].student_self_overlap_1_0.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_10",
            linestyle="dashdot"
        )
        dfs[feature].student_self_overlap_1_1.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_11",
            linestyle="solid"
        )
    plt.title("Self-Overlap")
    # plt.legend()
    fig.savefig(os.path.join(save_path, f"{save_name}_self_overlap.pdf"))

    # student-teacher0 overlap, R
    fig = plt.figure()
    for i, feature in enumerate(sorted_paths):
        label_base = feature.split("feature_")[1].split(f"_{varying_attribute}")[0]
        dfs[feature].student_teacher_0_overlap_0_0  .dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_00",
            linestyle="dashed"
        )
        dfs[feature].student_teacher_0_overlap_1_0.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_10",
            linestyle="dotted"
        )
    plt.title("Student-Teacher-0-Overlap")
    # plt.legend()
    fig.savefig(os.path.join(save_path, f"{save_name}_student_0_overlap.pdf"))

    # student-teacher1 overlap, T
    fig = plt.figure()
    for i, feature in enumerate(sorted_paths):
        label_base = feature.split("feature_")[1].split(f"_{varying_attribute}")[0]
        dfs[feature].student_teacher_1_overlap_0_0.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_00",
            linestyle="dashed"
        )
        dfs[feature].student_teacher_1_overlap_1_0.dropna().plot(
            color=color_map_1(i / len(sorted_paths)),
            label=f"{label_base}_10",
            linestyle="dotted"
        )
    plt.title("Student-Teacher-1-Overlap")
    # plt.legend()
    fig.savefig(os.path.join(save_path, f"{save_name}_student_1_overlap.pdf"))


if __name__ == "__main__":
    args = parser.parse_args()

    paths = [f for f in os.listdir(args.results_folder) if f.startswith("feature_")]

    attribute_values = []

    for i in args.attribute_values.strip("[").strip("]").split(","):
        try:
            value = int(i)
        except ValueError:
            value = "None"
        attribute_values.append(value)

    split_paths = {
        attribute_value: [p for p in paths if p.split("_")[-1] == str(attribute_value)]
        for attribute_value in attribute_values
    }

    split_dfs = {
        split: {
            f: pd.read_csv(os.path.join(args.results_folder, f, "0", "ode_log.csv"))
            for f in paths
        }
        for split, paths in split_paths.items()
    }

    for attribute_value, dfs in split_dfs.items():
        # plot_error_trajectories(
        #     dfs, args.varying_attribute, args.results_folder, attribute_value
        # )
        # plot_overlap_trajectories(
        #     dfs, args.varying_attribute, args.results_folder, attribute_value
        # )
        SWITCH_STEP = 1500000
        for t in [1600000, 1800000]:
            plot_cross_section(
                dfs, SWITCH_STEP, t, args.varying_attribute, args.results_folder, attribute_value
            )

    # if args.include is not None:
    #     included_experiments = [
    #         name.strip() for name in args.include.strip("[").strip("]").split(",")
    #     ]
    #     exp_names = included_experiments

    #     for name in included_experiments:
    #         assert name in os.listdir(
    #             args.results_folder
    #         ), f"name in include: {name} not in experiment names."
    # else:
    #     with open(config_changes_json_path, "r") as f:
    #         changes = json.load(f)
    #         exp_names = list(changes.keys())

    #     if args.exclude is not None:
    #         excluded_experiments = [
    #             name.strip() for name in args.exclude.strip("[").strip("]").split(",")
    #         ]

    #         exp_names = [name for name in exp_names if name not in excluded_experiments]

    # plotting_functions.plot_all_multi_seed_multi_run(
    #     folder_path=args.results_folder,
    #     exp_names=exp_names,
    #     window_width=args.smoothing,
    #     linewidth=args.linewidth,
    #     colormap=args.cmap,
    # )
