from typing import Any
from typing import Dict

import numpy as np

import constants
from ode import configuration
from ode import dynamics
from run import student_teacher_config


class ODERunner:
    """Runner for ode simulations."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        network_configuration: Dict[str, Any],
    ) -> None:
        self._config = config

    def run(self):
        if self._config.implementation == constants.Constants.CPP:
            self._run_cpp_ode()
        elif self._config.implementation == constants.Constants.PYTHON:
            self._run_python_ode(
                network_configuration=network_configuration,
                timestep=self._config.timestep,
            )
        else:
            raise ValueError(
                f"Implementation type {config.implementation} not recognised."
            )

    def _run_cpp_ode(self):
        raise NotImplementedError

    def _run_python_ode(self, network_configuration: Dict[str, Any], timestep: float):
        ode_configuration = configuration.StudentTwoTeacherConfiguration(
            Q=network_configuration[constants.Constants.STUDENT_SELF_OVERLAP],
            R=network_configuration[constants.Constants.STUDENT_TEACHER_OVERLAPS][0],
            U=network_configuration[constants.Constants.STUDENT_TEACHER_OVERLAPS][1],
            T=network_configuration[constants.Constants.TEACHER_SELF_OVERLAP][0],
            S=network_configuration[constants.Constants.TEACHER_SELF_OVERLAP][1],
            V=network_configuration[constants.Constants.TEACHER_CROSS_OVERLAPS][0],
            h1=network_configuration[constants.Constants.STUDENT_HEAD_WEIGHTS][0],
            h2=network_configuration[constants.Constants.STUDENT_HEAD_WEIGHTS][1],
            th1=network_configuration[constants.Constants.TEACHER_HEAD_WEIGHTS][0],
            th2=network_configuration[constants.Constants.TEACHER_HEAD_WEIGHTS][1],
        )

        curriculum = (
            np.arange(0, self._config.total_training_steps, self._config.fixed_period)[
                1:
            ]
            / self._config.input_dimension
        )

        ode = dynamics.StudentTeacherODE(
            overlap_configuration=ode_configuration,
            nonlinearity=self._config.student_nonlinearity,
            w_learning_rate=self._config.learning_rate,
            h_learning_rate=self._config.learning_rate,
            dt=timestep,
            curriculum=curriculum,
            soft_committee=self._config.soft_committee,
            train_first_layer=self._config.train_first_layer,
            train_head_layer=self._config.train_head_layer,
        )

        ode.step(self._config.total_training_steps / self._config.input_dimension)

        ode.save_to_csv(save_path=self._config.checkpoint_path)
        ode.make_plot(
            save_path=self._config.checkpoint_path,
            total_time=self._config.total_training_steps,
        )
