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


class MockModelChildClass(Model):
    """
    Mock implementation of Model base class

    Minimal implementations of the following methods are provided
    to override abstract base class methods:

    _construct_output_layers
    _output_forward
    """
    def __init__(self, config: Parameters, model_type: str):

        Model.__init__(
            self, config=config, model_type=model_type
            )

    def _construct_output_layers(self):
        return None

    def _output_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ModelTest(unittest.TestCase):
    """
    Test class for models.networks.base_network.Model class.
    Since Model is an abstract base class, a 'mock' class
    MockModelChildClass is used instead.
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

    def test_core_init(self):

        incorrect_model_types = ['teacher1', '', 'student_1']

        for incorrect_model_type in incorrect_model_types:
            self.assertRaises(
                    AssertionError, MockModelChildClass, self.sample_config,
                    incorrect_model_type
                )

    def test_student_nonlinearity(self):
        """
        This method tests that the correct nonlinearity functions
        are instantiated for the Model class when the child is a student.
        """
        base_config = self.sample_config

        relu_student_config = copy.deepcopy(base_config)
        sigmoid_student_config = copy.deepcopy(base_config)
        linear_student_config = copy.deepcopy(base_config)

        relu_student_config._config["model"]["student_nonlinearity"] = \
            'relu'
        self.assertEqual(
            relu_student_config.get(["model", "student_nonlinearity"]), 'relu'
            )

        sigmoid_student_config._config["model"]["student_nonlinearity"] = \
            'sigmoid'
        self.assertEqual(
            sigmoid_student_config.get(["model", "student_nonlinearity"]),
            'sigmoid'
            )

        linear_student_config._config["model"]["student_nonlinearity"] = \
            'linear'
        self.assertEqual(
            linear_student_config.get(["model", "student_nonlinearity"]),
            'linear'
            )

        relu_student = MockModelChildClass(relu_student_config, 'student')
        sigmoid_student = MockModelChildClass(
            sigmoid_student_config, 'student'
            )
        linear_student = MockModelChildClass(linear_student_config, 'student')

        self.assertEqual(relu_student.nonlinear_function, F.relu)
        self.assertEqual(sigmoid_student.nonlinear_function, torch.sigmoid)
        self.assertEqual(linear_student.nonlinear_function, linear_function)

    def test_teacher_nonlinearities(self):
        """
        This method tests that the correct nonlinearity functions
        are instantiated for the Model class when the child is a teacher.
        Individual teachers are instantiated
        with an element of a list specified in the config.
        """
        base_config = self.sample_config

        def _modify_config(new_teacher_nonlinearities: List) -> Parameters:
            modified_config = copy.deepcopy(base_config)
            modified_config._config["model"]["teacher_nonlinearities"] = \
                new_teacher_nonlinearities
            self.assertEqual(
                modified_config.get(["model", "teacher_nonlinearities"]),
                new_teacher_nonlinearities
                )
            return modified_config

        relu_relu_config = _modify_config(['relu', 'relu'])
        relu_sigmoid_config = _modify_config(['relu', 'sigmoid'])
        relu_linear_config = _modify_config(['relu', 'linear'])
        sigmoid_relu_config = _modify_config(['sigmoid', 'relu'])
        sigmoid_sigmoid_config = _modify_config(['sigmoid', 'sigmoid'])
        sigmoid_linear_config = _modify_config(['sigmoid', 'linear'])
        linear_relu_config = _modify_config(['linear', 'relu'])
        linear_sigmoid_config = _modify_config(['linear', 'sigmoid'])
        linear_linear_config = _modify_config(['linear', 'linear'])

        relu_relu_0 = MockModelChildClass(relu_relu_config, 'teacher_0')
        relu_relu_1 = MockModelChildClass(relu_relu_config, 'teacher_1')
        self.assertEqual(relu_relu_0.nonlinear_function, F.relu)
        self.assertEqual(relu_relu_1.nonlinear_function, F.relu)

        relu_sigmoid_0 = MockModelChildClass(relu_sigmoid_config, 'teacher_0')
        relu_sigmoid_1 = MockModelChildClass(relu_sigmoid_config, 'teacher_1')
        self.assertEqual(relu_sigmoid_0.nonlinear_function, F.relu)
        self.assertEqual(relu_sigmoid_1.nonlinear_function, torch.sigmoid)

        relu_linear_0 = MockModelChildClass(relu_linear_config, 'teacher_0')
        relu_linear_1 = MockModelChildClass(relu_linear_config, 'teacher_1')
        self.assertEqual(relu_linear_0.nonlinear_function, F.relu)
        self.assertEqual(relu_linear_1.nonlinear_function, linear_function)

        sigmoid_relu_0 = MockModelChildClass(sigmoid_relu_config, 'teacher_0')
        sigmoid_relu_1 = MockModelChildClass(sigmoid_relu_config, 'teacher_1')
        self.assertEqual(sigmoid_relu_0.nonlinear_function, torch.sigmoid)
        self.assertEqual(sigmoid_relu_1.nonlinear_function, F.relu)

        sigmoid_sigmoid_0 = MockModelChildClass(
            sigmoid_sigmoid_config, 'teacher_0'
            )
        sigmoid_sigmoid_1 = MockModelChildClass(
            sigmoid_sigmoid_config, 'teacher_1'
            )
        self.assertEqual(sigmoid_sigmoid_0.nonlinear_function, torch.sigmoid)
        self.assertEqual(sigmoid_sigmoid_1.nonlinear_function, torch.sigmoid)

        sigmoid_linear_0 = MockModelChildClass(
            sigmoid_linear_config, 'teacher_0'
            )
        sigmoid_linear_1 = MockModelChildClass(
            sigmoid_linear_config, 'teacher_1'
            )
        self.assertEqual(sigmoid_linear_0.nonlinear_function, torch.sigmoid)
        self.assertEqual(sigmoid_linear_1.nonlinear_function, linear_function)

        linear_relu_0 = MockModelChildClass(linear_relu_config, 'teacher_0')
        linear_relu_1 = MockModelChildClass(linear_relu_config, 'teacher_1')
        self.assertEqual(linear_relu_0.nonlinear_function, linear_function)
        self.assertEqual(linear_relu_1.nonlinear_function, F.relu)

        linear_linear_0 = MockModelChildClass(
            linear_linear_config, 'teacher_0'
            )
        linear_linear_1 = MockModelChildClass(
            linear_linear_config, 'teacher_1'
            )
        self.assertEqual(linear_linear_0.nonlinear_function, linear_function)
        self.assertEqual(linear_linear_1.nonlinear_function, linear_function)

        linear_sigmoid_0 = MockModelChildClass(
            linear_sigmoid_config, 'teacher_0'
            )
        linear_sigmoid_1 = MockModelChildClass(
            linear_sigmoid_config, 'teacher_1'
            )
        self.assertEqual(linear_sigmoid_0.nonlinear_function, linear_function)
        self.assertEqual(linear_sigmoid_1.nonlinear_function, torch.sigmoid)

    def test_construct_layers(self):
        """
        This method tests the layer construction, which is part of
        the model class initiatisation.

        Specifically:

        - Ensures layers attribute is correct type
        - Ensures lengths of layers attribute is correct
        """

        student_model = MockModelChildClass(self.sample_config, 'student')
        teacher_model = MockModelChildClass(self.sample_config, 'teacher_1')

        student_hidden_layers = \
            self.sample_config.get(["model", "student_hidden_layers"])
        teacher_hidden_layers = \
            self.sample_config.get(["model", "teacher_hidden_layers"])

        self.assertEqual(
            len(student_model.layers), len(student_hidden_layers)
            )
        self.assertEqual(
            len(teacher_model.layers), len(teacher_hidden_layers)
            )

        self.assertIsInstance(student_model.layers, torch.nn.ModuleList)
        self.assertIsInstance(teacher_model.layers, torch.nn.ModuleList)

    def test_weight_initialisation(self):
        """
        This method tests the normal distribution weight initialisation
        of the Model class. Behaviour is independent of student or teacher
        so arbitrarily we choose a 'student' model type here.
        """
        base_config = self.sample_config
        # make model large so initialisation std is representative
        base_config._config["model"]["input_dimension"] = 1000000

        zero_dev_config = copy.deepcopy(base_config)
        zero_dev_config._config["model"]["student_initialisation_std"] = 0

        low_dev_config = copy.deepcopy(base_config)
        low_dev_config._config["model"]["student_initialisation_std"] = 0.1

        large_dev_config = copy.deepcopy(base_config)
        large_dev_config._config["model"]["student_initialisation_std"] = 1

        self.assertRaises(
            RuntimeError, MockModelChildClass, zero_dev_config, 'student'
            )
        low_dev_mock_model = MockModelChildClass(low_dev_config, 'student')
        large_dev_mock_model = MockModelChildClass(large_dev_config, 'student')

        self.assertEqual(low_dev_mock_model.initialisation_std, 0.1)
        self.assertEqual(large_dev_mock_model.initialisation_std, 1)

        low_dev_layer = low_dev_mock_model.state_dict()['layers.0.weight']
        large_dev_layer = large_dev_mock_model.state_dict()['layers.0.weight']

        self.assertAlmostEqual(float(torch.std(low_dev_layer)), 0.1, places=2)
        self.assertAlmostEqual(float(torch.std(large_dev_layer)), 1, places=2)

    def test_forward(self):
        """
        This method tests the forward call of the Model class. Since this
        method makes a call to the _output_forward method, which is
        implemented by the child, we have an identity function for this
        method in our mock class.
        """
        base_config = self.sample_config

        mock_model = MockModelChildClass(base_config, 'student')

        input_dimension = base_config.get(["model", "input_dimension"])

        # ensure only correct dimensions are permitted
        incorrect_dimensions = [(0, 100), (25, 500), (5, 5000)]
        for incorrect_dimension in incorrect_dimensions:
            self.assertRaises(
                RuntimeError, mock_model.forward,
                torch.zeros(incorrect_dimension)
            )

        # check simple case of zero output
        zero_tensor_input = torch.zeros((5, input_dimension))
        zero_tensor_output = mock_model(zero_tensor_input)

        self.assertEqual(float(zero_tensor_output.sum()), 0)


def get_suite():
    model_tests = [
        'test_construct_layers', 'test_core_init', 'test_student_nonlinearity',
        'test_teacher_nonlinearities', 'test_weight_initialisation',
        'test_forward'
        ]
    return unittest.TestSuite(map(ModelTest, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=3)
runner.run(get_suite())
