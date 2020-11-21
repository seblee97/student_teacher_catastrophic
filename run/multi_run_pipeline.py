import argparse
import copy
import itertools
import os
from multiprocessing import Process
from typing import Any
from typing import List
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import pandas as pd

import constants
from run import student_teacher_config
from run import config_template
from run.config_changes import ConfigChange
from run import core_runner
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
    parser.add_argument("--seeds", metavar="-S", default=range(1))
    parser.add_argument(
        "--config_changes", metavar="-CC", default=ConfigChange.config_changes
    )

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

    for config_change in config_changes:
        run_name, changes = config_change.items()
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
    indices = os.listdir(folder)
    for index in indices:
        index_dfs = []
        for seed in seeds:
            index_dfs.append(pd.read_csv(f"{folder}/{index}/{seed}/ode_log.csv"))
        dfs[index] = index_dfs
    return dfs


def summary_plot(
    config: student_teacher_config.StudentTeacherConfiguration,
    experiment_path: str,
    seeds: List[int],
):
    # plot generalisation error over time for each overlap
    dfs = get_dfs(folder=experiment_path, seeds=seeds)

    teacher_1_fig = plt.figure()
    teacher_1_colormap = cm.get_cmap(constants.Constants.VIRIDIS)

    for i, index in enumerate(dfs.keys()):
        # TODO: ignore seeds for now!
        log_generalisation_error_0 = dfs[index][0][
            f"{constants.Constants.LOG_GENERALISATION_ERROR}_0"
        ].to_numpy()
        plt.plot(log_generalisation_error_0, color=teacher_1_colormap(i / len(dfs)))

    save_name = os.path.join(experiment_path, constants.Constants.FORGETTING_PLOT)
    teacher_1_fig.savefig(save_name, dpi=100)

    teacher_2_fig = plt.figure()
    teacher_2_colormap = cm.get_cmap(constants.Constants.PLASMA)

    for i, index in enumerate(dfs.keys()):
        # TODO: ignore seeds for now!
        log_generalisation_error_1 = dfs[index][0][
            f"{constants.Constants.LOG_GENERALISATION_ERROR}_1"
        ].to_numpy()
        plt.plot(log_generalisation_error_1, color=teacher_2_colormap(i / len(dfs)))

    save_name = os.path.join(experiment_path, constants.Constants.TRANSFER_PLOT)
    teacher_2_fig.savefig(save_name, dpi=100)


if __name__ == "__main__":

    args = get_args()
    base_configuration = get_config_object(args.config)
    base_configuration = set_device(config=base_configuration)

    timestamp = experiment_utils.get_experiment_timestamp()
    results_folder = os.path.join(MAIN_FILE_PATH, constants.Constants.RESULTS)
    experiment_path = os.path.join(results_folder, timestamp)

    if args.mode == constants.Constants.PARALLEL:
        parallel_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )
    elif args.mode == constants.Constants.SERIAL:
        serial_run(
            base_configuration=base_configuration,
            config_changes=args.config_changes,
            seeds=args.seeds,
            experiment_path=experiment_path,
            results_folder=results_folder,
            timestamp=timestamp,
        )

    summary_plot(
        config=base_configuration, experiment_path=experiment_path, seeds=args.seeds
    )
