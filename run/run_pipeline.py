import argparse
import datetime
import os
import time

import constants

from run import student_teacher_config
from run.config_template import ConfigTemplate
from run import core_runner

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args() -> argparse.Namespace:
    """Get args from command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-config",
        type=str,
        help="path to configuration file for student teacher experiment",
        default="config.yaml",
    )

    args = parser.parse_args()

    return args


def get_config_object(
    args: argparse.Namespace,
) -> student_teacher_config.StudentTeacherConfiguration:
    """Read config path into configuration object.

    Args:
        args: argparser namespace object with arguments specifying configuration.

    Returns:
        configuration: configuration object.
    """
    full_config_path = os.path.join(MAIN_FILE_PATH, args.config)
    configuration = student_teacher_config.StudentTeacherConfiguration(
        config=full_config_path, template=ConfigTemplate.base_config_template
    )
    return configuration


def set_random_seeds(seed: int) -> None:
    # import packages with non-deterministic behaviour
    import random

    import numpy as np
    import torch

    # set random seeds for these packages
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_experiment_metadata(
    config: student_teacher_config.StudentTeacherConfiguration,
) -> student_teacher_config.StudentTeacherConfiguration:
    """Set metadata (time, date, names) etc. for experiment."""
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    experiment_name = config.experiment_name or ""

    results_folder_base = constants.Constants.RESULTS

    checkpoint_path = (
        f"{MAIN_FILE_PATH}/{results_folder_base}/{exp_timestamp}/{experiment_name}/"
    )

    os.makedirs(checkpoint_path, exist_ok=True)

    config.add_property(constants.Constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(constants.Constants.EXPERIMENT_TIMESTAMP, exp_timestamp)

    return config


def run(config: student_teacher_config.StudentTeacherConfiguration):
    runner = core_runner.CoreRunner(config=config)
    runner.run()
    runner.post_process()


if __name__ == "__main__":
    args = get_args()
    config = get_config_object(args)
    config = set_experiment_metadata(config=config)
    set_random_seeds(seed=config.seed)
    run(config)
