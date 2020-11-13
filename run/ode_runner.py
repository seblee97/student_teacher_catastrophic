from typing import Any
from typing import Dict

import numpy as np

import constants
from ode import configuration
from ode import dynamics
from run import student_teacher_config

from utils import network_configuration
from utils import logger


class ODERunner:
    """Runner for ode simulations."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        network_configuration: network_configuration.NetworkConfiguration,
    ) -> None:
        self._config = config
        self._network_configuration = network_configuration

        if self._config.implementation == constants.Constants.PYTHON:
            self._logger = logger.Logger(
                config=self._config, csv_file_name=constants.Constants.ODE_CSV
            )

    def run(self):
        if self._config.implementation == constants.Constants.CPP:
            self._run_cpp_ode()
        elif self._config.implementation == constants.Constants.PYTHON:
            self._run_python_ode(
                network_configuration=self._network_configuration,
                timestep=self._config.timestep,
            )
        else:
            raise ValueError(
                f"Implementation type {self._config.implementation} not recognised."
            )

    def _run_cpp_ode(self):
        raise NotImplementedError

    def _run_python_ode(self, network_configuration: Dict[str, Any], timestep: float):
        ode_configuration = configuration.StudentTwoTeacherConfiguration(
            Q=self._network_configuration.student_self_overlap,
            R=self._network_configuration.student_teacher_overlaps[0],
            U=self._network_configuration.student_teacher_overlaps[1],
            T=self._network_configuration.teacher_self_overlaps[0],
            S=self._network_configuration.teacher_self_overlaps[1],
            V=self._network_configuration.teacher_cross_overlaps[0],
            h1=self._network_configuration.student_head_weights[0],
            h2=self._network_configuration.student_head_weights[1],
            th1=self._network_configuration.teacher_head_weights[0],
            th2=self._network_configuration.teacher_head_weights[1],
        )

        curriculum = (
            np.arange(0, self._config.total_training_steps, self._config.fixed_period)[
                1:
            ]
            / self._config.input_dimension
        )

        time = self._config.total_training_steps / self._config.input_dimension

        ode = dynamics.StudentTeacherODE(
            overlap_configuration=ode_configuration,
            nonlinearity=self._config.student_nonlinearity,
            w_learning_rate=self._config.learning_rate,
            h_learning_rate=self._config.learning_rate,
            dt=timestep,
            curriculum=curriculum,
            soft_committee=self._config.soft_committee,
            train_first_layer=self._config.train_hidden_layers,
            train_head_layer=self._config.train_head_layer,
            frozen_feature=self._config.frozen_feature,
        )

        steps = 0
        while ode.time < time:
            if steps % self._config.checkpoint_frequency == 0 and steps != 0:
                self._logger.checkpoint_df()
            if ode.time > ode.next_switch_step:
                ode.switch_teacher()
            ode.step()
            self._logger.log_generalisation_errors(
                step=steps, generalisation_errors=[ode.error_1, ode.error_2]
            )
            if self._config.log_overlaps and steps % self._config.log_frequency == 0:
                self._logger.log_network_configuration(
                    step=steps, network_configuration=ode.configuration
                )
            steps += 1

        self._logger.checkpoint_df()

        # ode.save_to_csv(save_path=self._config.checkpoint_path)
        # ode.make_plot(
        #     save_path=self._config.checkpoint_path,
        #     total_time=self._config.total_training_steps,
        # )
