import argparse
import copy
import itertools
import os
from multiprocessing import Process
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm

import constants
from run import config_template
from run import core_runner
from run import student_teacher_config
from run.config_changes import ConfigChange
from utils import experiment_utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args() -> argparse.Namespace:
    """Get args from command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        metavar="-M",
        default="parallel",
        help="run in 'parallel' or 'serial'",
    )
    parser.add_argument("--config", metavar="-C", default="config.yaml")
    parser.add_argument("--seeds", metavar="-S", default="[0]")
    parser.add_argument(
        "--config_changes", metavar="-CC", default=ConfigChange.config_changes
    )
    parser.add_argument("--skip_summary", action="store_true")

    args = parser.parse_args()

    return args


def get_config_object(
    config: str,
) -> student_teacher_config.StudentTeacherConfiguration:
    """Read config path into configuration object.

    Args:
        config: path to configuration file.

    Returns:
        configuration: configuration object.
    """
    full_config_path = os.path.join(MAIN_FILE_PATH, args.config)
    configuration = student_teacher_config.StudentTeacherConfiguration(
        config=full_config_path,
        template=config_template.ConfigTemplate.base_config_template,
    )
    return configuration


def set_device(
    config: student_teacher_config.StudentTeacherConfiguration,
) -> student_teacher_config.StudentTeacherConfiguration:
    """Establish availability of GPU."""
    if config.use_gpu:
        print("Attempting to find GPU...")
        if torch.cuda.is_available():
            print("GPU found, using the GPU...")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            config.add_property(constants.Constants.USING_GPU, True)
            experiment_device = torch.device("cuda:{}".format(config.gpu_id))
        else:
            print("GPU not found, reverting to CPU")
            config.add_property(constants.Constants.USING_GPU, False)
            experiment_device = torch.device("cpu")
    else:
        print("Using the CPU")
        experiment_device = torch.device("cpu")
    config.add_property(constants.Constants.EXPERIMENT_DEVICE, experiment_device)
    return config


def parallel_run(
    base_configuration: student_teacher_config.StudentTeacherConfiguration,
    seeds: List[int],
    config_changes: Dict[str, List[Tuple[str, Any]]],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    procs = []

    for run_name, changes in config_changes.items():
        for seed in seeds:
            p = Process(
                target=single_run,
                args=(
                    base_configuration,
                    run_name,
                    seed,
                    results_folder,
                    experiment_path,
                    timestamp,
                    changes,
                ),
            )
            p.start()
            procs.append(p)

    for p in procs:
        p.join()


def serial_run(
    base_configuration: student_teacher_config.StudentTeacherConfiguration,
    seeds: List[int],
    config_changes: Dict[str, List[Tuple[str, Any]]],
    experiment_path: str,
    results_folder: str,
    timestamp: str,
):
    for run_name, changes in config_changes.items():
        print(f"{run_name}")
        for seed in seeds:
            print(f"Seed: {seed}")
            single_run(
                base_configuration=base_configuration,
                seed=seed,
                results_folder=results_folder,
                experiment_path=experiment_path,
                timestamp=timestamp,
                run_name=run_name,
                config_change=changes,
            )


def single_run(
    base_configuration: student_teacher_config.StudentTeacherConfiguration,
    run_name: str,
    seed: int,
    results_folder: str,
    experiment_path: str,
    timestamp: str,
    config_change: List[Tuple[str, Any]],
):
    config = copy.deepcopy(base_configuration)

    experiment_utils.set_random_seeds(seed)
    checkpoint_path = experiment_utils.get_checkpoint_path(
        results_folder, timestamp, run_name, str(seed)
    )

    config.amend_property(
        property_name=constants.Constants.SEED, new_property_value=seed
    )

    for change in config_change:
        config.amend_property(
            property_name=change[0],
            new_property_value=change[1],
        )

    config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, timestamp)
    config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)

    r = core_runner.CoreRunner(config=config)

    r.run()
    r.post_process()


def get_dfs(folder: str, seeds: List[int]) -> Dict[str, List[pd.DataFrame]]:
    dfs = {}
    indices = [index for index in os.listdir(folder) if not index.startswith(".")]
    for index in indices:
        index_dfs = []
        for seed in seeds:
            index_dfs.append(pd.read_csv(f"{folder}/{index}/{seed}/ode_log.csv"))
        dfs[index] = index_dfs
    return dfs


def generalisation_error_figs(
    dfs: List[pd.DataFrame], indices: List[str], num_steps: int, seeds: List[int]
) -> None:
    """Plot generalisation error over time for each overlap."""
    teacher_1_fig = plt.figure()
    teacher_1_colormap = cm.get_cmap(constants.Constants.VIRIDIS)

    for i, index in enumerate(indices):
        log_generalisation_error_0s = []
        for seed in seeds:
            log_generalisation_error_0 = dfs[index][seed][
                f"{constants.Constants.LOG_GENERALISATION_ERROR}_0"
            ].to_numpy()
            log_generalisation_error_0s.append(log_generalisation_error_0)

        scaling = num_steps / len(log_generalisation_error_0)
        mean_log_generalisation_error_0 = np.mean(log_generalisation_error_0s, axis=0)
        std_log_generalisation_error_0 = np.std(log_generalisation_error_0s, axis=0)
        plt.plot(
            scaling * np.arange(len(mean_log_generalisation_error_0)),
            mean_log_generalisation_error_0,
            color=teacher_1_colormap(i / len(dfs)),
            label=index,
        )
        plt.fill_between(
            scaling * np.arange(len(mean_log_generalisation_error_0)),
            mean_log_generalisation_error_0 - std_log_generalisation_error_0,
            mean_log_generalisation_error_0 + std_log_generalisation_error_0,
            color=teacher_1_colormap(i / len(dfs)),
            alpha=0.3,
            label=index,
        )

    if len(indices) > 5:
        pass
    else:
        plt.legend()
    save_name = os.path.join(experiment_path, constants.Constants.FORGETTING_PLOT)
    teacher_1_fig.savefig(save_name, dpi=100)
    plt.close()

    teacher_2_fig = plt.figure()
    teacher_2_colormap = cm.get_cmap(constants.Constants.PLASMA)

    for i, index in enumerate(indices):
        log_generalisation_error_1s = []
        for seed in seeds:
            log_generalisation_error_1 = dfs[index][seed][
                f"{constants.Constants.LOG_GENERALISATION_ERROR}_1"
            ].to_numpy()
            log_generalisation_error_1s.append(log_generalisation_error_1)

        scaling = num_steps / len(log_generalisation_error_1)
        mean_log_generalisation_error_1 = np.mean(log_generalisation_error_1s, axis=0)
        std_log_generalisation_error_1 = np.std(log_generalisation_error_1s, axis=0)
        plt.plot(
            scaling * np.arange(len(mean_log_generalisation_error_1)),
            mean_log_generalisation_error_1,
            color=teacher_2_colormap(i / len(dfs)),
            label=index,
            alpha=0.3,
        )
        plt.fill_between(
            scaling * np.arange(len(mean_log_generalisation_error_1)),
            mean_log_generalisation_error_1 - std_log_generalisation_error_1,
            mean_log_generalisation_error_1 + std_log_generalisation_error_1,
            color=teacher_2_colormap(i / len(dfs)),
            label=index,
            alpha=0.3,
        )

    if len(indices) > 5:
        pass
    else:
        plt.legend()
    save_name = os.path.join(experiment_path, constants.Constants.TRANSFER_PLOT)
    teacher_2_fig.savefig(save_name, dpi=100)
    plt.close()


def cross_section_figs(
    dfs: List[pd.DataFrame],
    indices: List[str],
    seeds: List[int],
    switch_step: int,
    num_ode_steps: int,
):
    """Plot forgetting/transfer vs. v for various time intervals post switch."""
    overlaps = [float(index.split("_")[1]) for index in indices]

    for interval in np.linspace(switch_step, num_ode_steps, 10)[:-1]:

        error_deltas_0_means = []
        error_deltas_0_stds = []
        error_deltas_1_means = []
        error_deltas_1_stds = []

        for i, index in enumerate(indices):

            error_deltas_0 = []
            error_deltas_1 = []

            for seed in seeds:
                generalisation_error_0 = dfs[index][seed][
                    f"{constants.Constants.GENERALISATION_ERROR}_0"
                ].to_numpy()
                generalisation_error_1 = dfs[index][seed][
                    f"{constants.Constants.GENERALISATION_ERROR}_1"
                ].to_numpy()
                switch_error_0 = generalisation_error_0[switch_step]
                switch_error_1 = generalisation_error_1[switch_step]
                error_delta_0 = generalisation_error_0[int(interval)] - switch_error_0
                error_delta_1 = switch_error_1 - generalisation_error_1[int(interval)]

                error_deltas_0.append(error_delta_0)
                error_deltas_1.append(error_delta_1)

            error_deltas_0_means.append(np.mean(error_deltas_0))
            error_deltas_0_stds.append(np.std(error_deltas_0))
            error_deltas_1_means.append(np.mean(error_deltas_1))
            error_deltas_1_stds.append(np.std(error_deltas_1))

        forgetting_vs_v_fig = plt.figure()
        plt.plot(overlaps, error_deltas_0_means, linewidth=5, color="b")
        plt.fill_between(
            overlaps,
            np.array(error_deltas_0_means) - np.array(error_deltas_0_stds),
            np.array(error_deltas_0_means) + np.array(error_deltas_0_stds),
            linewidth=5,
            color="b",
            alpha=0.3,
        )
        plt.xlabel("Overlap")
        plt.ylabel(f"Forgetting {int(interval) - switch_step} Steps Post-Switch")
        save_name = os.path.join(
            experiment_path,
            f"{int(interval) - switch_step}_{constants.Constants.FORGETTING_VS_V_PLOT}",
        )
        forgetting_vs_v_fig.savefig(save_name, dpi=100)
        plt.close()

        transfer_vs_v_fig = plt.figure()
        plt.plot(overlaps, error_deltas_1_means, linewidth=5, color="r")
        plt.fill_between(
            overlaps,
            np.array(error_deltas_1_means) - np.array(error_deltas_1_stds),
            np.array(error_deltas_1_means) + np.array(error_deltas_1_stds),
            linewidth=5,
            color="r",
            alpha=0.3,
        )
        plt.xlabel("Overlap")
        plt.ylabel(f"Transfer {int(interval) - switch_step} Steps Post-Switch")
        save_name = os.path.join(
            experiment_path,
            f"{int(interval) - switch_step}_{constants.Constants.TRANSFER_VS_V_PLOT}",
        )
        transfer_vs_v_fig.savefig(save_name, dpi=100)
        plt.close()


def rate_figs(
    dfs: List[pd.DataFrame],
    indices: List[str],
    switch_step: int,
    seeds: List[int],
    num_points: int,
):
    """Plot initial forgetting/transfer rate vs. v."""

    forgetting_rates_mean = []
    forgetting_rates_std = []
    transfer_rates_mean = []
    transfer_rates_std = []

    overlaps = [float(index.split("_")[1]) for index in indices]

    for i, index in enumerate(indices):

        error_delta_rate_0s = []
        error_delta_rate_1s = []

        for seed in seeds:
            generalisation_error_0 = dfs[index][seed][
                f"{constants.Constants.GENERALISATION_ERROR}_0"
            ].to_numpy()
            generalisation_error_1 = dfs[index][seed][
                f"{constants.Constants.GENERALISATION_ERROR}_1"
            ].to_numpy()

            initial_error_deltas_0 = [
                generalisation_error_0[switch_step + i + 1]
                - generalisation_error_0[switch_step + i]
                for i in range(num_points)
            ]
            initial_error_deltas_1 = [
                generalisation_error_1[switch_step + i]
                - generalisation_error_1[switch_step + i + 1]
                for i in range(num_points)
            ]

            error_delta_rate_0 = np.mean(initial_error_deltas_0)
            error_delta_rate_1 = np.mean(initial_error_deltas_1)

            error_delta_rate_0s.append(error_delta_rate_0)
            error_delta_rate_1s.append(error_delta_rate_1)

        forgetting_rates_mean.append(np.mean(error_delta_rate_0s))
        forgetting_rates_std.append(np.std(error_delta_rate_0s))

        transfer_rates_mean.append(np.mean(error_delta_rate_1s))
        transfer_rates_std.append(np.std(error_delta_rate_1s))

    forgetting_rate_fig = plt.figure()
    plt.plot(overlaps, forgetting_rates_mean, linewidth=5, color="b")
    plt.fill_between(
        overlaps,
        np.array(forgetting_rates_mean) - np.array(forgetting_rates_std),
        np.array(forgetting_rates_mean) + np.array(forgetting_rates_std),
        linewidth=5,
        color="b",
        alpha=0.3,
    )
    plt.xlabel("Overlap")
    plt.ylabel("Initial Rate of Forgetting")
    save_name = os.path.join(
        experiment_path,
        f"{num_points}_{constants.Constants.FORGETTING_RATE_PLOT}",
    )
    forgetting_rate_fig.savefig(save_name, dpi=100)
    plt.close()

    transfer_rate_fig = plt.figure()
    plt.plot(overlaps, transfer_rates_mean, linewidth=5, color="r")
    plt.fill_between(
        overlaps,
        np.array(transfer_rates_mean) - np.array(transfer_rates_std),
        np.array(transfer_rates_mean) + np.array(transfer_rates_std),
        linewidth=5,
        color="r",
        alpha=0.3,
    )
    plt.xlabel("Overlap")
    plt.ylabel("Initial Rate of Transfer")
    save_name = os.path.join(
        experiment_path,
        f"{num_points}_{constants.Constants.TRANSFER_RATE_PLOT}",
    )
    transfer_rate_fig.savefig(save_name, dpi=100)
    plt.close()


def summary_plot(
    config: student_teacher_config.StudentTeacherConfiguration,
    experiment_path: str,
    seeds: List[int],
):
    num_steps = config.total_training_steps

    if config.ode_simulation:
        num_ode_steps = (
            config.total_training_steps / config.input_dimension / config.timestep
        )
        # scale step for ODE time
        switch_step = int(config.switch_steps[0] / config.input_dimension / config.timestep)
    else:
        num_ode_steps = num_steps
        switch_step = int(config.switch_steps[0])


    dfs = get_dfs(folder=experiment_path, seeds=seeds)
    indices = sorted(dfs.keys(), key=lambda x: float(x.split("_")[1]))

    generalisation_error_figs(
        dfs=dfs, indices=indices, num_steps=num_steps, seeds=seeds
    )
    cross_section_figs(
        dfs=dfs,
        indices=indices,
        switch_step=switch_step,
        num_ode_steps=num_ode_steps,
        seeds=seeds,
    )
    rate_figs(
        dfs=dfs, indices=indices, switch_step=switch_step, seeds=seeds, num_points=10
    )
    rate_figs(
        dfs=dfs, indices=indices, switch_step=switch_step, seeds=seeds, num_points=100
    )
    rate_figs(
        dfs=dfs, indices=indices, switch_step=switch_step, seeds=seeds, num_points=200
    )


if __name__ == "__main__":

    args = get_args()
    seeds = [int(s) for s in args.seeds.strip("[").strip("]").split(",")]
    base_configuration = get_config_object(args.config)
    base_configuration = set_device(config=base_configuration)

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    if args.mode == constants.Constants.PARALLEL:
        parallel_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    elif args.mode == constants.Constants.SERIAL:
        serial_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )

    if not args.skip_summary:
        summary_plot(
            config=base_configuration, experiment_path=experiment_path, seeds=seeds
        )
