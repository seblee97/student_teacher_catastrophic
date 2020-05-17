import unittest

from abc import ABC, abstractmethod

import os
import yaml
import pandas as pd
import numpy as np
import shutil

from experiments.student_teacher_runner import StudentTeacherRunner
from experiments.student_teacher_parameters import StudentTeacherParameters
from experiments.config_templates import ConfigTemplate, MNISTDataTemplate, \
    TrainedMNISTTemplate, PureMNISTTemplate

FILE_PATH = os.path.dirname(os.path.realpath(__file__))


class TestRunner(ABC, unittest.TestCase):

    @abstractmethod
    def setUp(self):
        raise NotImplementedError

    def tearDown(self):
        # remove files saved during 'test' training
        shutil.rmtree(os.path.join(FILE_PATH, 'test_results/test/'))

    def _finalise_config(self):

        checkpoint_path = os.path.join(FILE_PATH, 'test_results/test/')

        self.sample_config.set_property("checkpoint_path", checkpoint_path)
        self.sample_config.set_property("experiment_timestamp", None)

        # get specified random seed value from config
        seed_value = self.sample_config.get("seed")

        # import packages with non-deterministic behaviour
        import random
        import numpy as np
        import torch
        # set random seeds for these packages
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)

        self.sample_config.set_property("device", torch.device("cpu"))

        log_path = '{}data_logger.csv'.format(checkpoint_path)
        self.sample_config.set_property("logfile_path", log_path)

    def test_train(self):

        print("Training 'mock' model. This can take some time...")

        self.runner.train()
        self.runner.logger._consolidate_dfs()

        log_df = pd.read_csv(
            os.path.join(FILE_PATH, "test_results/test/data_logger.csv")
        )

        num_steps = \
            self.sample_config.get(["training", "total_training_steps"])

        attributes = ["training_loss", "mean_generalisation_error/log"]

        def _numpy_test(array_1, array_2):
            np.testing.assert_almost_equal(
                array_1, array_2, 5
            )
            return True

        for attribute in attributes:
            log_df_attribute = np.array(log_df[attribute].tolist()[:num_steps])
            target_df_attribute = np.array(
                self.target_df[attribute].tolist()[:num_steps]
            )
            self.assertEqual(
                _numpy_test(log_df_attribute, target_df_attribute),
                True
                )


class IIDRunner(TestRunner):

    IID_MODEL_YAML_PATH = os.path.join(
        FILE_PATH, "iid_gaussian_overlapping.yaml"
        )

    IID_TARGET_DF_PATH = os.path.join(
        FILE_PATH, "target_dfs/data_logger.csv"
        )

    def setUp(self):

        with open(self.IID_MODEL_YAML_PATH, 'r') as yaml_file:
            params = yaml.load(yaml_file, yaml.SafeLoader)

        self.sample_config = StudentTeacherParameters(
            params, root_config_template=ConfigTemplate,
            mnist_data_config_template=MNISTDataTemplate,
            trained_mnist_config_template=TrainedMNISTTemplate,
            pure_mnist_config_template=PureMNISTTemplate
        )

        self._finalise_config()

        self.runner = StudentTeacherRunner(self.sample_config)
        self.target_df = pd.read_csv(self.IID_TARGET_DF_PATH)


def get_suite():
    model_tests = [
        'test_train'
        ]
    return unittest.TestSuite(map(IIDRunner, model_tests))


runner = unittest.TextTestRunner(buffer=True, verbosity=3)
runner.run(get_suite())
