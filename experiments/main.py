
from context import model, utils

import argparse
import torch
import yaml
import time
import datetime
import os

parser = argparse.ArgumentParser()

parser.add_argument('-config', type=str, help='path to configuration file for student teacher experiment', default='config.yaml')
parser.add_argument('-gpu_id', type=int, help='id of gpu to use if more than 1 available', default=0)

args = parser.parse_args()

if __name__ == "__main__":

    # read parameters from config
    with open(args.base_config, 'r') as base_yaml_file:
        base_params = yaml.load(base_yaml_file, yaml.SafeLoader)

    # create object in which to store experiment parameters
    student_teacher_parameters = utils.parameters.StudentTeacherParameters(base_params)

    # establish experiment name / log path etc.
    exp_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = student_teacher_parameters.get("experiment_name")
    if experiment_name:
        checkpoint_path = 'results/{}/{}/'.format(exp_timestamp, experiment_name)
    else:
        checkpoint_path = 'results/{}/'.format(exp_timestamp)
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
        