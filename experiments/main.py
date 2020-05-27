from utils import Argparser

from postprocessing import StudentTeacherPostprocessor
from experiments.student_teacher_runner import StudentTeacherRunner
from experiments.student_teacher_parameters import StudentTeacherParameters
from experiments.config_templates import ConfigTemplate, MNISTDataTemplate, \
    TrainedMNISTTemplate, PureMNISTTemplate

import argparse
import yaml
import time
import datetime
import os

from typing import Dict

MAIN_FILE_PATH = os.path.dirname(os.path.realpath(__file__))


def post_process(args):

    post_process_args = {}
    if args.crop_start:
        post_process_args["crop_start"] = args.crop_start
    if args.crop_end:
        post_process_args["crop_end"] = args.crop_end
    if args.pfn:
        post_process_args["figure_name"] = args.pfn

    post_process_args["combine_plots"] = not args.ipf
    post_process_args["show_legends"] = not args.nl
    post_process_args["repeats"] = args.repeats

    post_processor = StudentTeacherPostprocessor(
        save_path=args.ppp,
        extra_args=post_process_args
        )
    post_processor.post_process()


def run(args):

    # read base-parameters from base-config
    base_config_full_path = os.path.join(MAIN_FILE_PATH, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    params = Argparser.update_config_with_parser(args=args, params=params)

    # create object in which to store experiment parameters and validate
    # config file
    student_teacher_parameters = StudentTeacherParameters(
        params, root_config_template=ConfigTemplate,
        mnist_data_config_template=MNISTDataTemplate,
        trained_mnist_config_template=TrainedMNISTTemplate,
        pure_mnist_config_template=PureMNISTTemplate
        )

    # establish experiment name / log path etc.
    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = student_teacher_parameters.get("experiment_name")

    if args.cp:
        results_folder_base = args.cp if args.cp.endswith('/') \
            else args.cp + '/'
    else:
        results_folder_base = 'results/'

    if experiment_name:
        checkpoint_path = '{}/{}/{}/{}/'.format(
            MAIN_FILE_PATH, results_folder_base, exp_timestamp,
            experiment_name
            )
    else:
        checkpoint_path = '{}/{}/{}/'.format(
            MAIN_FILE_PATH, results_folder_base, exp_timestamp
            )

    student_teacher_parameters.set_property("checkpoint_path", checkpoint_path)
    student_teacher_parameters.set_property(
        "experiment_timestamp", exp_timestamp
        )

    # get specified random seed value from config
    seed_value = student_teacher_parameters.get("seed")

    # import packages with non-deterministic behaviour
    import random
    import numpy as np
    import torch
    # set random seeds for these packages
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # establish whether gpu is available
    if torch.cuda.is_available() and student_teacher_parameters.get('use_gpu'):
        print("Using the GPU")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        student_teacher_parameters.set_property("using_gpu", True)
        experiment_device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        print("Using the CPU")
        student_teacher_parameters.set_property("using_gpu", False)
        experiment_device = torch.device("cpu")

    if args.log_ext:
        log_path = '{}data_logger.csv'.format(checkpoint_path)
        student_teacher_parameters.set_property("logfile_path", log_path)

    # write copy of config_yaml in model_checkpoint_folder
    student_teacher_parameters.save_configuration(checkpoint_path)

    student_teacher_parameters.set_property("device", experiment_device)

    student_teacher = StudentTeacherRunner(config=student_teacher_parameters)
    student_teacher.train()

    if args.app:
        post_processor = StudentTeacherPostprocessor(
            save_path=checkpoint_path, plot_config_path=args.pcp)
        post_processor.post_process()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = Argparser.process_parser(parser)

    if args.ppp is not None:
        post_process(args)
    else:
        run(args)
