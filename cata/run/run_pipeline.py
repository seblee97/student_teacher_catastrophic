import argparse
import datetime
import os
import time

import constants
import torch
from run import core_runner
from run import student_teacher_config
from run.config_template import ConfigTemplate
from utils import experiment_utils

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def get_args() -> argparse.Namespace:
    """Get args from command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="path to configuration file for student teacher experiment",
        default="config.yaml",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="seed to be used",
        default=None,
    )

    parser.add_argument(
        "--fa",
        type=float,
        help="feature rotation alpha",
        default=None,
    )

    parser.add_argument(
        "--ra",
        type=float,
        help="readout rotation alpha",
        default=None,
    )

    parser.add_argument(
        "--name",
        type=str,
        help="name of experiment",
        default=None,
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        help="id of GPU to use",
        default=None,
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

    if args.seed is not None:
        configuration.amend_property(property_name="seed", new_property_value=args.seed)
    if args.fa is not None:
        configuration.amend_property(
            property_name="feature_rotation_alpha", new_property_value=args.fa
        )
    if args.ra is not None:
        configuration.amend_property(
            property_name="readout_rotation_alpha", new_property_value=args.ra
        )
    if args.name is not None:
        configuration.amend_property(
            property_name="experiment_name", new_property_value=args.name
        )
    if args.gpu_id is not None:
        configuration.amend_property(
            property_name="gpu_id", new_property_value=args.gpu_id
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

    results_folder_base = config.results_path or MAIN_FILE_PATH
    results_folder_base = os.path.join(results_folder_base, constants.RESULTS)

    checkpoint_path = experiment_utils.get_checkpoint_path(
        folder=results_folder_base,
        timestamp=exp_timestamp,
        experiment_name=experiment_name,
    )

    os.makedirs(checkpoint_path, exist_ok=True)

    config.add_property(constants.CHECKPOINT_PATH, checkpoint_path)
    config.add_property(constants.EXPERIMENT_TIMESTAMP, exp_timestamp)

    return config


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
            config.add_property(constants.USING_GPU, True)
            experiment_device = torch.device("cuda:{}".format(config.gpu_id))
        else:
            print("GPU not found, reverting to CPU")
            config.add_property(constants.USING_GPU, False)
            #experiment_device = torch.device("cpu")
            experiment_device = torch.device('mps')
    else:
        print("Using the CPU")
        #experiment_device = torch.device("cpu")
        experiment_device = torch.device('mps')
    config.add_property(constants.EXPERIMENT_DEVICE, experiment_device)
    return config


def run(config: student_teacher_config.StudentTeacherConfiguration):
    runner = core_runner.CoreRunner(config=config)
    runner.run()
    runner.post_process()


if __name__ == "__main__":
    args = get_args()
    config = get_config_object(args)
    config = set_experiment_metadata(config=config)
    config = set_device(config=config)
    set_random_seeds(seed=config.seed)
    run(config)
