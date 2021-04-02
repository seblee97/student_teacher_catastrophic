import copy
import os
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List

import numpy as np
import torch
import torch.nn as nn

from constants import Constants
from curricula import base_curriculum
from curricula import hard_steps_curriculum
from curricula import periodic_curriculum
from curricula import threshold_curriculum
from data_modules import base_data_module
from data_modules import iid_data
from loggers import base_logger
from loggers import split_logger
from loggers import unified_logger
from run import student_teacher_config
from students import base_student
from students import continual_student
from students import meta_student
from teachers.ensembles import base_teacher_ensemble
from teachers.ensembles import both_rotation_ensemble
from teachers.ensembles import feature_rotation_ensemble
from teachers.ensembles import readout_rotation_ensemble
from utils import decorators
from utils import experiment_utils
from utils import network_configuration


class NetworkRunner:
    """Runner for network simulations.

    Class for orchestrating student teacher framework including training
    and test loops.
    """
    def __init__(
            self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> None:
        """
        Class constructor.

        Args:
            config: configuration object containing parameters to specify run.
        """
        # extract class-relevant attributes from config
        self._seed = config.seed
        self._device = config.experiment_device
        self._input_dimension = config.input_dimension
        self._checkpoint_frequency = config.checkpoint_frequency
        self._checkpoint_path = config.checkpoint_path
        self._save_weight_frequency = config.save_weight_frequency
        self._total_training_steps = config.total_training_steps
        self._log_frequency = config.log_frequency
        self._test_frequency = config.test_frequency
        self._total_step_count = 0
        self._log_overlaps = config.log_overlaps

        # initialise student, teachers, logger_module,
        # data_module, loss_module, torch optimiser, and curriculum object
        self._teachers = self._setup_teachers(config=config)
        self._student = self._setup_student(config=config)
        self._logger = self._setup_logger(config=config)
        self._data_module = self._setup_data(config=config)
        self._loss_function = self._setup_loss(config=config)
        self._optimiser = self._setup_optimiser(config=config)
        self._curriculum = self._setup_curriculum(config=config)

        self._manage_network_devices()

    def get_network_configuration(
            self) -> network_configuration.NetworkConfiguration:
        """Get macroscopic configuration of networks in terms of order parameters.

        Used for both logging purposes and as input to ODE runner.
        """
        with torch.no_grad():
            student_head_weights = [
                head.weight.data.cpu().numpy().flatten()
                for head in self._student.heads
            ]
            teacher_head_weights = [
                teacher.head.weight.data.cpu().numpy().flatten()
                for teacher in self._teachers.teachers
            ]
            student_self_overlap = self._student.self_overlap.cpu().numpy()
            teacher_self_overlaps = [
                teacher.self_overlap.cpu().numpy()
                for teacher in self._teachers.teachers
            ]
            teacher_cross_overlaps = [
                o.cpu().numpy() for o in self._teachers.cross_overlaps
            ]
            student_layer = self._student.layers[0].weight.data
            student_teacher_overlaps = [
                student_layer.mm(
                    teacher.layers[0].weight.data.t()).cpu().numpy() /
                self._input_dimension for teacher in self._teachers.teachers
            ]

        return network_configuration.NetworkConfiguration(
            student_head_weights=student_head_weights,
            teacher_head_weights=teacher_head_weights,
            student_self_overlap=student_self_overlap,
            teacher_self_overlaps=teacher_self_overlaps,
            teacher_cross_overlaps=teacher_cross_overlaps,
            student_teacher_overlaps=student_teacher_overlaps,
        )

    @decorators.timer
    def _setup_teachers(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_teacher_ensemble.BaseTeacherEnsemble:
        """Initialise teacher object containing teacher networks."""
        forward_scaling = (1 / np.sqrt(config.teacher_hidden_layers[0])
                           if config.scale_teacher_forward_by_hidden else 1.0)
        base_arguments = {
            Constants.INPUT_DIMENSION: config.input_dimension,
            Constants.HIDDEN_DIMENSIONS: config.teacher_hidden_layers,
            Constants.OUTPUT_DIMENSION: config.output_dimension,
            Constants.BIAS: config.teacher_bias_parameters,
            Constants.NUM_TEACHERS: config.num_teachers,
            Constants.LOSS_TYPE: config.loss_type,
            Constants.NONLINEARITY: config.student_nonlinearity,
            Constants.SCALE_HIDDEN_LR: config.scale_hidden_lr,
            Constants.FORWARD_SCALING: forward_scaling,
            Constants.UNIT_NORM_TEACHER_HEAD: config.unit_norm_teacher_head,
            Constants.INITIALISATION_STD: config.teacher_initialisation_std,
        }
        if config.teacher_configuration == Constants.FEATURE_ROTATION:
            teachers_class = feature_rotation_ensemble.FeatureRotationTeacherEnsemble
            additional_arguments = {
                Constants.ROTATION_MAGNITUDE: config.feature_rotation_magnitude
            }
        elif config.teacher_configuration == Constants.READOUT_ROTATION:
            teachers_class = readout_rotation_ensemble.ReadoutRotationTeacherEnsemble
            additional_arguments = {
                Constants.ROTATION_MAGNITUDE:
                config.readout_rotation_magnitude,
                Constants.FEATURE_COPY_PERCENTAGE:
                config.feature_copy_percentage,
            }
        elif config.teacher_configuration == Constants.BOTH_ROTATION:
            teachers_class = both_rotation_ensemble.BothRotationTeacherEnsemble
            additional_arguments = {
                Constants.FEATURE_ROTATION_ALPHA:
                config.feature_rotation_alpha,
                Constants.READOUT_ROTATION_ALPHA:
                config.readout_rotation_alpha,
            }
        else:
            raise ValueError(
                f"Teacher configuration '{config.teacher_configuration}' not recognised."
            )

        teachers = teachers_class(**base_arguments, **additional_arguments)

        if config.save_teacher_weights:
            save_path = os.path.join(config.checkpoint_path,
                                     Constants.TEACHER_WEIGHT_SAVE_PATH)
            teachers.save_all_teacher_weights(save_path=save_path)

        return teachers

    @decorators.timer
    def _setup_student(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_student.BaseStudent:
        """Initialise object containing student network."""
        if config.learner_configuration == Constants.CONTINUAL:
            student_class = continual_student.ContinualStudent
        elif config.learner_configuration == Constants.META:
            student_class = meta_student.MetaStudent
        else:
            raise ValueError(
                f"Learner type '{config.learning_configuration}' not recognised"
            )

        teacher_features_copy = (
            None if config.teacher_features_copy is None else copy.deepcopy(
                self._teachers.teachers[config.teacher_features_copy].layers))

        return student_class(
            input_dimension=config.input_dimension,
            hidden_dimensions=config.student_hidden_layers,
            output_dimension=config.output_dimension,
            bias=config.student_bias_parameters,
            soft_committee=config.soft_committee,
            num_teachers=config.num_teachers,
            loss_type=config.loss_type,
            learning_rate=config.learning_rate,
            scale_head_lr=config.scale_head_lr,
            scale_hidden_lr=config.scale_hidden_lr,
            scale_forward_by_hidden=config.scale_student_forward_by_hidden,
            nonlinearity=config.student_nonlinearity,
            freeze_features=config.freeze_features,
            train_hidden_layers=config.train_hidden_layers,
            train_head_layer=config.train_head_layer,
            initialise_outputs=config.initialise_student_outputs,
            apply_nonlinearity_on_output=config.apply_nonlinearity_on_output,
            symmetric_initialisation=config.symmetric_student_initialisation,
            initialisation_std=config.student_initialisation_std,
            teacher_features_copy=teacher_features_copy,
        )

    @decorators.timer
    def _setup_logger(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_logger.BaseLogger:
        if config.split_logging:
            logger = split_logger.SplitLogger(
                config=config,
                run_type=Constants.SIM,
                network_config=self.get_network_configuration(),
            )
        else:
            logger = unified_logger.UnifiedLogger(
                config=config,
                run_type=Constants.SIM,
            )
        return logger

    @decorators.timer
    def _setup_data(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_data_module.BaseData:
        """Initialise data module."""
        if config.input_source == Constants.IID_GAUSSIAN:
            data_module = iid_data.IIDData(
                train_batch_size=config.train_batch_size,
                test_batch_size=config.test_batch_size,
                input_dimension=config.input_dimension,
                mean=config.mean,
                variance=config.variance,
                dataset_size=config.dataset_size,
            )
        else:
            raise ValueError(
                f"Data module (specified by input source) {config.input_source} not recognised"
            )
        return data_module

    @decorators.timer
    def _setup_loss(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> Callable:
        """Instantiate torch loss function"""
        if config.loss_function == "mse":
            loss_function = nn.MSELoss()
        elif config.loss_function == "bce":
            loss_function = nn.BCELoss()
        else:
            raise NotImplementedError(
                f"Loss function {config.loss_function} not recognised.")
        return loss_function

    @decorators.timer
    def _setup_curriculum(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if config.stopping_condition == Constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=config)
        elif config.stopping_condition == Constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(
                config=config)
        elif config.stopping_condition == Constants.SWITCH_STEPS:
            curriculum = hard_steps_curriculum.HardStepsCurriculum(
                config=config)
        else:
            raise ValueError(
                f"Stopping condition {config.stopping_condition} not recognised."
            )
        return curriculum

    @decorators.timer
    def _setup_optimiser(
        self, config: student_teacher_config.StudentTeacherConfiguration
    ) -> torch.optim.SGD:
        """Initialise optimiser with trainable parameters of student."""
        trainable_parameters = self._student.get_trainable_parameters()
        return torch.optim.SGD(trainable_parameters, lr=config.learning_rate)

    @decorators.timer
    def _manage_network_devices(self) -> None:
        """Move relevant networks etc. to device specified in config (CPU or GPU)."""
        self._student.to(device=self._device)
        for teacher in self._teachers.teachers:
            teacher.to(device=self._device)

    def _compute_loss(self, prediction: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
        """Calculate loss of prediction of student vs. target from teacher

        Args:
            prediction: prediction made by student network on given input
            target: teacher output on same input

        Returns:
            loss: loss between target (from teacher) and prediction (from student)
        """
        loss = 0.5 * self._loss_function(prediction, target)
        return loss

    @decorators.timer
    def _setup_training(self):
        """Prepare runner for training, including constructing a test dataset.
        This method must be called before training loop is called.
        """
        # different configurations make different number of calls to rng
        # reset seeds before training to ensure data is the same.
        experiment_utils.set_random_seeds(self._seed)

        self._test_data_inputs = self._data_module.get_test_data()[
            Constants.X].to(self._device)
        self._test_teacher_outputs = self._teachers.forward_all(
            self._test_data_inputs)

    def train(self):
        """Training orchestration."""

        self._setup_training()

        while self._total_step_count < self._total_training_steps:
            teacher_index = next(self._curriculum)

            self._train_on_teacher(teacher_index=teacher_index)

        self._logger.checkpoint_df()

    def _train_on_teacher(self, teacher_index: int):
        """One phase of training (wrt one teacher)."""
        self._student.signal_task_boundary(new_task=teacher_index)
        task_step_count = 0
        generalisation_errors = self._compute_generalisation_errors()
        self._logger.log_generalisation_errors(
            step=self._total_step_count,
            generalisation_errors=generalisation_errors,
        )
        timer = time.time()

        while self._total_step_count < self._total_training_steps:

            if (self._total_step_count % self._checkpoint_frequency == 0
                    and self._total_step_count != 0):
                self._logger.checkpoint_df()

            if self._total_step_count % self._test_frequency == 0:
                generalisation_errors = self._compute_generalisation_errors()
                self._logger.log_generalisation_errors(
                    step=self._total_step_count,
                    generalisation_errors=generalisation_errors,
                )
            if self._log_overlaps and self._total_step_count % self._log_frequency == 0:
                self._logger.log_network_configuration(
                    step=self._total_step_count,
                    network_config=self.get_network_configuration(),
                )

            if self._total_step_count % 500 == 0:
                print(
                    f"Generalisation errors @ step {self._total_step_count} "
                    f"({task_step_count}'th step training on teacher {teacher_index}): "
                )
                if self._total_step_count != 0:
                    print(f"Time for last 500 steps: {time.time() - timer}")
                    timer = time.time()
                for i, error in enumerate(generalisation_errors):
                    print(f"    Teacher {i}: {error}\n")

            latest_task_generalisation_error = generalisation_errors[
                teacher_index]

            self._training_step(teacher_index=teacher_index)

            task_step_count += 1

            if self._curriculum.to_switch(
                    task_step=task_step_count,
                    error=latest_task_generalisation_error):
                break

    def _training_step(self, teacher_index: int):
        """Perform single training step."""

        if self._save_weight_frequency is not None:
            if self._total_step_count % self._save_weight_frequency == 0:
                self._student.save_weights(save_path=os.path.join(
                    self._checkpoint_path,
                    f"{Constants.STUDENT_WEIGHTS}_{self._total_step_count}"
                ))

        batch = self._data_module.get_batch()
        batch_input = batch[Constants.X].to(self._device)

        # forward through student network
        student_output = self._student.forward(batch_input)

        # forward through teacher network(s)
        teacher_output = self._teachers.forward(teacher_index, batch_input)

        # training iteration
        self._optimiser.zero_grad()
        loss = self._compute_loss(student_output, teacher_output)
        loss.backward()
        self._optimiser.step()

        self._total_step_count += 1

        self._student.signal_step(step=self._total_step_count)

    def _compute_generalisation_errors(self) -> List[float]:
        """Compute test errors for student with respect to all teachers."""
        self._student.eval()

        generalisation_errors = []

        with torch.no_grad():
            student_outputs = self._student.forward_all(self._test_data_inputs)

            for student_output, teacher_output in zip(
                    student_outputs, self._test_teacher_outputs):
                loss = self._compute_loss(student_output, teacher_output)
                generalisation_errors.append(loss.item())

        self._student.train()

        return generalisation_errors
