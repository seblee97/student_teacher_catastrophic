from context import utils
from student_teacher_runner import StudentTeacherRunner

import argparse
import torch
import yaml
import time
import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument('-config', type=str, help='path to configuration file for student teacher experiment', default='base_config.yaml')
parser.add_argument('-additional_config', '--ac', type=str, help='path to folder contatining supplementary configuration files', default='configs/')
parser.add_argument('-gpu_id', type=int, help='id of gpu to use if more than 1 available', default=0)
parser.add_argument('-log_ext', action='store_false', help='whether to write evaluation data to external file as well as tensorboard')

parser.add_argument('-seed', '--s', type=int, help='seed to use for packages with prng', default=None)
parser.add_argument('-learner_configuration', '--lc', type=str, help="meta or continual", default=None)
parser.add_argument('-teacher_configuration', '--tc', type=str, help="noisy or independent or mnist", default=None)
parser.add_argument('-input_source', '--inp_s', type=str, help="mnist or iid_gaussian", default=None)
parser.add_argument('-input_dim', '--id', type=int, help="input dimension to networks", default=None)
parser.add_argument('-num_teachers', '--nt', type=int, default=None)
parser.add_argument('-loss_type', '--lty', type=str, default=None)
parser.add_argument('-loss_function', '--lf', type=str, default=None)
parser.add_argument('-selection_type', '--st', type=str, help="random or cyclical", default=None)
parser.add_argument('-stopping_condition', '--sc', type=str, help="threshold or fixed_period", default=None)
parser.add_argument('-fixed_period', '--fp', type=int, help="time between teacher change", default=None)
parser.add_argument('-loss_threshold', '--lt', type=str, help="how low loss for current teacher goes before switching (used with threshold)", default=None)
parser.add_argument('-student_nonlinearity', '--snl', type=str, help="which non linearity to use for student", default=None)
parser.add_argument('-teacher_nonlinearities', '--tnl', type=str, help="which non linearity to use for teacher", default=None)
parser.add_argument('-teacher_hidden', '--th', type=str, help="dimension of hidden layer in teacher", default=None)
parser.add_argument('-student_hidden', '--sh', type=str, help="dimension of hidden layer in student", default=None)
parser.add_argument('-learning_rate', '--lr', type=float, help="base learning rate", default=None)
parser.add_argument('-total_steps', '--ts', type=int, help="total timesteps to run algorithm", default=None)
parser.add_argument('-experiment_name', '--en', type=str, help="name to give to experiment", default=None)
parser.add_argument('-verbose', '--v', type=int, help="whether to display prints", default=None)
parser.add_argument('-checkpoint_path', '--cp', type=str, help="where to log results", default=None)
parser.add_argument('-checkpoint_frequency', '--cf', type=float, help="how often to log results", default=None) 
parser.add_argument('-teacher_overlaps', '--to', type=str, help="per layer overlaps between teachers. Must be in format '[30, 20, etc.]'", default=None)

args = parser.parse_args()

if __name__ == "__main__":

    main_file_path = os.path.dirname(os.path.realpath(__file__))

    # read base-parameters from base-config
    base_config_full_path = os.path.join(main_file_path, args.config)
    with open(base_config_full_path, 'r') as yaml_file:
        params = yaml.load(yaml_file, yaml.SafeLoader)

    # create object in which to store experiment parameters
    student_teacher_parameters = utils.parameters.StudentTeacherParameters(params)

    # update parameters with (optional) args given in command line
    if args.s:
        student_teacher_parameters._config["seed"] = args.s
    if args.lc:
        student_teacher_parameters._config["task"]["learner_configuration"] = args.lc
    if args.tc:
        student_teacher_parameters._config["task"]["teacher_configuration"] = args.tc
    if args.inp_s:
        student_teacher_parameters._config["training"]["input_source"] = args.inp_s
    if args.id:
        student_teacher_parameters._config["model"]["input_dimension"] = args.id
    if args.nt:
        student_teacher_parameters._config["task"]["num_teachers"] = args.nt
    if args.lty:
        student_teacher_parameters._config["task"]["loss_type"] = args.lty
    if args.lf:
        student_teacher_parameters._config["training"]["loss_function"] = args.lf
    if args.st:
        student_teacher_parameters._config["curriculum"]["selection_type"] = args.st
    if args.sc:
        student_teacher_parameters._config["curriculum"]["stopping_condition"] = args.sc
    if args.fp:
        student_teacher_parameters._config["curriculum"]["fixed_period"] = args.fp
    if args.lt:
        student_teacher_parameters._config["curriculum"]["loss_threshold"] = args.lt
    if args.ts:
        student_teacher_parameters._config["training"]["total_training_steps"] = args.ts
    if args.en:
        student_teacher_parameters._config["experiment_name"] = args.en
    if args.lr:
        student_teacher_parameters._config["training"]["learning_rate"] = args.lr
    if args.cf:
        student_teacher_parameters._config["checkpoint_frequency"] = args.cf
    if args.v is not None:
        student_teacher_parameters._config["verbose"] = bool(args.v)

    supplementary_configs_path = args.ac
    additional_configurations = []

    teacher_configuration = student_teacher_parameters.get(["task", "teacher_configuration"])
    if teacher_configuration == 'noisy':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'noisy_config.yaml'))
    elif teacher_configuration == 'independent':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'independent_config.yaml'))
    elif teacher_configuration == 'drifting':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'drifting_config.yaml'))
    elif teacher_configuration == 'overlapping':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'overlapping_config.yaml'))
    
    elif teacher_configuration == "trained_mnist":
        additional_configurations.append(os.path.join(supplementary_configs_path, 'trained_mnist_config.yaml'))

    elif teacher_configuration == 'mnist':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'mnist_config.yaml'))
    else:
        raise ValueError("teacher configuration {} not recognised. Please use 'noisy', \
                'overlapping', 'drifting', 'independent', or 'mnist'".format(teacher_configuration))

    if student_teacher_parameters.get(["training", "input_source"]) == 'mnist':
        additional_configurations.append(os.path.join(supplementary_configs_path, 'mnist_input_config.yaml'))

    # specific parameters
    for additional_configuration in additional_configurations:
        additional_configuration_full_path = os.path.join(main_file_path, additional_configuration)
        with open(additional_configuration_full_path, 'r') as yaml_file:
            specific_params = yaml.load(yaml_file, yaml.SafeLoader)
    
        # update base-parameters with specific parameters
        student_teacher_parameters.update(specific_params)

    # update specific parameters with (optional) args given in command line
    if args.to:
        overlaps = [int(op) for op in "".join(args.to).strip('[]').split(',')]
        student_teacher_parameters._config["task"]["overlap_percentages"] = overlaps
    if args.snl:
        student_teacher_parameters._config["model"]["student_nonlinearity"] = args.snl
    if args.tnl:
        teacher_nonlinearities = [str(nl).strip() for nl in "".join(args.tnl).strip('[]').split(',')]
        student_teacher_parameters._config["model"]["teacher_nonlinearities"] = teacher_nonlinearities
    if args.sh:
        student_hidden = [int(h) for h in "".join(args.sh).strip('[]').split(',')]
        student_teacher_parameters._config["model"]["student_hidden_layers"] = student_hidden
    if args.th:
        teacher_hidden = [int(h) for h in "".join(args.th).strip('[]').split(',')]
        student_teacher_parameters._config["model"]["teacher_hidden_layers"] = teacher_hidden

    # check for consistency in loss specification of config
    permissible_regression_losses = ["mse", "l1", "smooth_l1"]
    permissible_classification_losses = ["bce"]

    loss_type = student_teacher_parameters.get(["task", "loss_type"])
    loss_function = student_teacher_parameters.get(["training", "loss_function"])

    if (loss_type == "classification" and loss_function in permissible_classification_losses) or \
       (loss_type == "regression" and loss_function in permissible_regression_losses):
       pass
    else:
        raise ValueError("Potential inconsistency in config specification for loss type and loss function. Please check")

    # establish experiment name / log path etc.
    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = student_teacher_parameters.get("experiment_name")

    if args.cp:
        results_folder_base = args.cp if args.cp.endswith('/') else args.cp + '/'
    else:
        results_folder_base = 'results/'
        
    if experiment_name:
        checkpoint_path = '{}/{}/{}/{}/'.format(main_file_path, results_folder_base, exp_timestamp, experiment_name)
    else:
        checkpoint_path = '{}/{}/{}/'.format(main_file_path, results_folder_base, exp_timestamp)

    student_teacher_parameters.set_property("checkpoint_path", checkpoint_path)
    student_teacher_parameters.set_property("experiment_timestamp", exp_timestamp)

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
        student_teacher_parameters.set_property("device", "cuda")
        experiment_device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        print("Using the CPU")
        student_teacher_parameters.set_property("device", "cpu")
        experiment_device = torch.device("cpu")

    # write copy of config_yaml in model_checkpoint_folder
    student_teacher_parameters.save_configuration(checkpoint_path)

    if args.log_ext:
        log_path = '{}data_logger.csv'.format(checkpoint_path)
        student_teacher_parameters.set_property("logfile_path", log_path)
    
    student_teacher = StudentTeacherRunner(config=student_teacher_parameters)
    student_teacher.train()
    