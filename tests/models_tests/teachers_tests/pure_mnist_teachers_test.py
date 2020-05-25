import unittest

import os
import yaml
import copy

import torch
import torch.nn.functional as F

from typing import List

from experiments.student_teacher_parameters import StudentTeacherParameters
from experiments.config_templates import ConfigTemplate, MNISTDataTemplate, \
    TrainedMNISTTemplate, PureMNISTTemplate

from models.networks.base_network import Model
from utils import Parameters, linear_function

file_path = os.path.dirname(os.path.realpath(__file__))
TEST_MODEL_YAML_PATH = os.path.join(file_path, "model_test_config.yaml")


class PureMNISTTeachersTest(unittest.TestCase):
    """
    Test class for models.teachers.overlapping_teachers class.
    """
    def setUp(self):

        with open(TEST_MODEL_YAML_PATH, 'r') as yaml_file:
            params = yaml.load(yaml_file, yaml.SafeLoader)

        self.sample_config = StudentTeacherParameters(
            params, root_config_template=ConfigTemplate,
            mnist_data_config_template=MNISTDataTemplate,
            trained_mnist_config_template=TrainedMNISTTemplate,
            pure_mnist_config_template=PureMNISTTemplate
        )


def get_suite():
    model_tests = [
        ]
    return unittest.TestSuite(map(PureMNISTTeachersTest, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=3)
runner.run(get_suite())
