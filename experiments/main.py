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


def process_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-config', type=str, help='path to configuration file for student \
            teacher experiment', default='base_config.yaml'
        )
    parser.add_argument(
        '-gpu_id', type=int, help='id of gpu to use if more than 1 available',
        default=0
        )
    parser.add_argument(
        '-log_ext', action='store_false', help='whether to write evaluation \
            data to external file as well as tensorboard'
        )
    parser.add_argument(
        '-plot_config_path', '--pcp', type=str, help='path to json for plot \
            config', default="plot_configs/summary_plots.json"
    )
    parser.add_argument(
        '-auto_post_process', '--app', action='store_false', help='whether to \
            automatically go into postprocessing after training loop'
    )
    parser.add_argument(
        '-post_processing_path', '--ppp', type=str, help='path to folder to \
            post-process', default=None
        )

    parser.add_argument(
        '-seed', '--s', type=int, help='seed to use for packages with prng',
        default=None
        )
    parser.add_argument(
        '-learner_configuration', '--lc', type=str, help="meta or continual",
        default=None
        )
    parser.add_argument(
        '-teacher_configuration', '--tc', type=str, help="noisy or \
            independent or mnist", default=None
        )
    parser.add_argument(
        '-input_source', '--inp_s', type=str, help="mnist or iid_gaussian",
        default=None)
    parser.add_argument(
        '-input_dim', '--id', type=int, help="input dimension to networks",
        default=None
        )
    parser.add_argument(
        '-num_teachers', '--nt', type=int, default=None
        )
    parser.add_argument(
        '-loss_type', '--lty', type=str, default=None
        )
    parser.add_argument(
        '-loss_function', '--lf', type=str, default=None
        )
    parser.add_argument(
        '-selection_type', '--st', type=str, help="random or cyclical",
        default=None
        )
    parser.add_argument(
        '-stopping_condition', '--sc', type=str, help="threshold or \
            fixed_period", default=None
        )
    parser.add_argument(
        '-fixed_period', '--fp', type=int, help="time between teacher change",
        default=None
        )
    parser.add_argument(
        '-loss_threshold', '--lt', type=str, help="how low loss for current \
            teacher goes before switching (used with threshold)", default=None
        )
    parser.add_argument(
        '-student_nonlinearity', '--snl', type=str, help="which non linearity \
            to use for student", default=None
        )
    parser.add_argument(
        '-teacher_nonlinearities', '--tnl', type=str, help="which non \
            linearity to use for teacher", default=None
        )
    parser.add_argument(
        '-teacher_hidden', '--th', type=str, help="dimension of hidden layer \
            in teacher", default=None
        )
    parser.add_argument(
        '-student_hidden', '--sh', type=str, help="dimension of hidden layer \
            in student", default=None
        )
    parser.add_argument(
        '-learning_rate', '--lr', type=float, help="base learning rate",
        default=None
        )
    parser.add_argument(
        '-total_steps', '--ts', type=int, help="total timesteps to run \
            algorithm", default=None
        )
    parser.add_argument(
        '-experiment_name', '--en', type=str, help="name to give to \
            experiment", default=None
        )
    parser.add_argument(
        '-verbose', '--v', type=int, help="whether to display prints",
        default=None
        )
    parser.add_argument(
        '-checkpoint_path', '--cp', type=str, help="where to log results",
        default=None
        )
    parser.add_argument(
        '-checkpoint_frequency', '--cf', type=float, help="how often to log \
            results", default=None)

    parser.add_argument(
        '-teacher_overlaps', '--to', type=str, help="per layer overlaps \
            between teachers. Must be in format '[30, 20, etc.]'", default=None
        )

    args = parser.parse_args()

    return args


def update_config_with_parser(args, params: Dict):

    # update parameters with (optional) args given in command line
    if args.s:
        params["seed"] = args.s
    if args.lc:
        params["task"]["learner_configuration"] = args.lc
    if args.tc:
        params["task"]["teacher_configuration"] = args.tc
    if args.inp_s:
        params["training"]["input_source"] = args.inp_s
    if args.id:
        params["model"]["input_dimension"] = args.id
    if args.nt:
        params["task"]["num_teachers"] = args.nt
    if args.lty:
        params["task"]["loss_type"] = args.lty
    if args.lf:
        params["training"]["loss_function"] = args.lf
    if args.st:
        params["curriculum"]["selection_type"] = args.st
    if args.sc:
        params["curriculum"]["stopping_condition"] = args.sc
    if args.fp:
        params["curriculum"]["fixed_period"] = args.fp
    if args.ts:
        params["training"]["total_training_steps"] = args.ts
    if args.en:
        params["experiment_name"] = args.en
    if args.lr:
        params["training"]["learning_rate"] = args.lr
    if args.cf:
        params["checkpoint_frequency"] = args.cf
    if args.v is not None:
        params["verbose"] = bool(args.v)

    # update specific parameters with (optional) args given in command line
    if args.to:
        overlaps = [int(op) for op in "".join(args.to).strip('[]').split(',')]
        params["teachers"]["overlap_percentages"] = overlaps
    if args.snl:
        params["model"]["student_nonlinearity"] = args.snl
    if args.tnl:
        teacher_nonlinearities = [
            str(nl).strip() for nl in "".join(args.tnl).strip('[]').split(',')
            ]
        params["model"]["teacher_nonlinearities"] = teacher_nonlinearities
    if args.sh:
        student_hidden = [
            int(h) for h in "".join(args.sh).strip('[]').split(',')
            ]
        params["model"]["student_hidden_layers"] = student_hidden
    if args.th:
        teacher_hidden = [
            int(h) for h in "".join(args.th).strip('[]').split(',')
            ]
        params["model"]["teacher_hidden_layers"] = teacher_hidden
    if args.lt:
        if isinstance(args.lt, float):
            params["curriculum"]["loss_threshold"] = args.lt
        else:
            threshold_sequence = [
                float(thr) for thr in "".join(args.lt).strip('[]').split(',')
            ]
            params["curriculum"]["loss_threshold"] = threshold_sequence

    return params


def postprocess(save_path: str, plot_config_path: str):

    full_plot_config_path = os.path.join(MAIN_FILE_PATH, plot_config_path)

    post_processor = StudentTeacherPostprocessor(
        save_path=args.ppp, plot_config_path=full_plot_config_path)
    post_processor.post_process()


def run(args):

    # read base-parameters from base-config
    base_config_full_path = os.path.join(MAIN_FILE_PATH, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    params = update_config_with_parser(args=args, params=params)

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

    args = process_parser()

    if args.ppp is not None:
        postprocess(save_path=args.ppp, plot_config_path=args.pcp)
    else:
        run(args)
