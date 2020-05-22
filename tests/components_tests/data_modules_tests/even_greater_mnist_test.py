import unittest

from abc import ABC, abstractmethod

import os
import yaml
import torch

from experiments.student_teacher_parameters import StudentTeacherParameters
from experiments.config_templates import ConfigTemplate, MNISTDataTemplate, \
    TrainedMNISTTemplate, PureMNISTTemplate

from components.data_modules.even_greater_mnist_data \
    import MNISTEvenGreaterData

from constants import Constants

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class _EvenGreaterMNISTTest(unittest.TestCase, ABC):
    """
    Test class for even_greater_mnist_data.MNISTEvenGreaterData
    in components.data_modules.
    """
    @abstractmethod
    def _get_config_path(self):
        raise NotImplementedError

    def setUp(self):
        config_path = self._get_config_path()

        test_model_yaml_path = \
            os.path.join(
                FILE_PATH, "even_greater_configs", config_path
                )

        with open(test_model_yaml_path, 'r') as yaml_file:
            params = yaml.load(yaml_file, yaml.SafeLoader)

        self.sample_config = StudentTeacherParameters(
            params, root_config_template=ConfigTemplate,
            mnist_data_config_template=MNISTDataTemplate,
            trained_mnist_config_template=TrainedMNISTTemplate,
            pure_mnist_config_template=PureMNISTTemplate
        )

        self.sample_config.set_property("device", torch.device("cpu"))

        self.even_greater_class = MNISTEvenGreaterData(self.sample_config)

    def test_init(self):
        """
        This method tests that the correct number of train and
        test iterators are instantiated.
        """
        self.assertEqual(
            len(self.even_greater_class.training_data_iterators), 2
            )
        self.assertEqual(
            len(self.even_greater_class.test_data_iterators), 2
            )

    def test_test_set_batching(self):
        """
        This method tests that the test set iterators produce the whole
        test dataset
        """
        for test_iterator in self.even_greater_class.test_data_iterators:
            test_batch = next(test_iterator)
            self.assertEqual(len(test_batch[0]), Constants.MNIST_TEST_SET_SIZE)
            self.assertRaises(StopIteration, next, test_iterator)

    def test_train_set_batching(self):
        """
        This method tests whether the training iterators are producing the
        correct batch sizes
        """
        train_batch_size = \
            self.sample_config.get(["training", "train_batch_size"])

        expected_num_batches = \
            Constants.MNIST_TRAIN_SET_SIZE // train_batch_size

        for train_iterator in self.even_greater_class.training_data_iterators:

            for _ in range(expected_num_batches):
                batch = next(train_iterator)
                self.assertEqual(len(batch[0]), train_batch_size)
                self.assertEqual(len(batch[1]), train_batch_size)

                self.assertEqual(
                    batch[0].shape,
                    (train_batch_size, Constants.MNIST_FLATTENED_DIM)
                    )
                self.assertEqual(len(batch[1]), 1)

            self.assertRaises(StopIteration, next, train_iterator)

    def test_get_test_data(self):
        """
        This method tests that the get_test_data method produces
        the whole test set and has the right return format
        """
        test_data = self.even_greater_class.get_test_data()

        self.assertIsInstance(test_data, dict)
        self.assertEqual(list(test_data.keys()), ['x', 'y'])

        self.assertIsInstance(test_data['x'], torch.Tensor)
        self.assertIsInstance(test_data['y'], list)
        self.assertEqual(len(test_data['y']), 2)

        self.assertEqual(
            test_data['x'].shape,
            (Constants.MNIST_TEST_SET_SIZE, Constants.MNIST_FLATTENED_DIM)
            )

        for labels in test_data['y']:
            self.assertIsInstance(labels, torch.Tensor)
            self.assertEqual(
                labels.shape,
                (Constants.MNIST_TEST_SET_SIZE, 1)
                )

    def _test_get_batch_from_task(self, task_index):
        """
        This method tests that the get_batch_method returns
        training batches of the correct dimensions and resets
        the iterator correctly after a cycle through the data.
        """
        self.even_greater_class.signal_task_boundary_to_data_generator(
            task_index
            )

        train_batch_size = \
            self.sample_config.get(["training", "train_batch_size"])

        expected_num_batches = \
            Constants.MNIST_TRAIN_SET_SIZE // train_batch_size

        for b in range(expected_num_batches):
            batch = self.even_greater_class.get_batch()

            self.assertEqual(len(batch['x']), train_batch_size)
            self.assertEqual(len(batch['y']), train_batch_size)

            self.assertEqual(
                batch['x'].shape,
                (train_batch_size, Constants.MNIST_FLATTENED_DIM)
                )
            self.assertEqual(batch['y'].shape, (1, 1))

        self.assertEqual(
            len(list(
                self.even_greater_class.training_data_iterators[task_index]
                )),
            Constants.MNIST_TRAIN_SET_SIZE
            - expected_num_batches * train_batch_size
            )

        next_cycle_batch = self.even_greater_class.get_batch()

        self.assertEqual(len(next_cycle_batch['x']), train_batch_size)
        self.assertEqual(len(next_cycle_batch['y']), train_batch_size)

        self.assertEqual(
            next_cycle_batch['x'].shape,
            (train_batch_size, Constants.MNIST_FLATTENED_DIM)
            )
        self.assertEqual(next_cycle_batch['y'].shape, (1, 1))

    def test_get_batch(self):

        for t in range(2):
            self._test_get_batch_from_task(t)


class EvenGreaterMNISTTest1(_EvenGreaterMNISTTest):

    def _get_config_path(self):
        return "even_greater_config_1.yaml"


def get_suite():
    model_tests = [
        'test_init', 'test_test_set_batching', 'test_train_set_batching',
        'test_get_test_data', 'test_get_batch'
        ]
    return unittest.TestSuite(map(EvenGreaterMNISTTest1, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=3)
runner.run(get_suite())
