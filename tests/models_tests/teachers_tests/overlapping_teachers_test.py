import unittest

from abc import ABC, abstractmethod

import os
import yaml

import torch

from experiments.student_teacher_parameters import StudentTeacherParameters
from experiments.config_templates import ConfigTemplate, MNISTDataTemplate, \
    TrainedMNISTTemplate, PureMNISTTemplate

from models.teachers.overlapping_teachers import OverlappingTeachers

file_path = os.path.dirname(os.path.realpath(__file__))
TEST_MODEL_YAML_PATH = os.path.join(file_path, "overlapping_test_config.yaml")


class OverlappingTeachersTest(unittest.TestCase, ABC):
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

        self.sample_config.set_property("device", torch.device("cpu"))

        self.complete_setUp()

    @abstractmethod
    def complete_setUp(self):
        raise NotImplementedError("Base class method")

    def test_init(self):

        expected_num_teachers = \
            self.sample_config.get(["task", "num_teachers"])

        self.assertEquals(
            len(self.overlapping_class._teachers),
            expected_num_teachers
            )

    def test_overlap(self):

        expected_overlaps = \
            self.sample_config.get(["teachers", "overlap_percentages"])

        # get teacher weights
        teacher_weights = [
            t.state_dict() for t in self.overlapping_class._teachers
            ]

        original_teacher_weights = teacher_weights[0]

        for weights in teacher_weights[1:]:
            for layer_index, layer in enumerate(weights):
                expected_copy_ratio = round(
                    0.01 * expected_overlaps[layer_index] * len(layer)
                    )
                if expected_copy_ratio == 0:
                    all_equal = torch.equal(
                        weights[layer][expected_copy_ratio:],
                        original_teacher_weights[layer][expected_copy_ratio:]
                    )
                    self.assertFalse(all_equal)
                elif expected_copy_ratio == 100:
                    all_equal = torch.equal(
                        weights[layer][:expected_copy_ratio],
                        original_teacher_weights[layer][:expected_copy_ratio]
                        )
                    self.assertTrue(all_equal)
                else:
                    all_equal = torch.equal(
                        weights[layer][:expected_copy_ratio],
                        original_teacher_weights[layer][:expected_copy_ratio]
                        )
                    self.assertTrue(all_equal)
                    all_equal = torch.equal(
                        weights[layer][expected_copy_ratio:],
                        original_teacher_weights[layer][expected_copy_ratio:]
                        )
                    self.assertFalse(all_equal)


class Overlapping1(OverlappingTeachersTest):

    def complete_setUp(self):
        self.overlapping_class = OverlappingTeachers(self.sample_config)


class Overlapping2(OverlappingTeachersTest):

    def complete_setUp(self):

        self.sample_config._config["teachers"]["overlaps"] = [100, 0]

        self.overlapping_class = OverlappingTeachers(self.sample_config)


class Overlapping3(OverlappingTeachersTest):

    def complete_setUp(self):

        self.sample_config._config["teachers"]["overlaps"] = [50, 50]

        self.overlapping_class = OverlappingTeachers(self.sample_config)


def get_suite(test_class):
    model_tests = [
        'test_init', 'test_overlap'
        ]
    return unittest.TestSuite(map(test_class, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=3)

all_test_classes = [Overlapping1, Overlapping2, Overlapping3]
for test_class in all_test_classes:
    runner.run(get_suite(test_class))
