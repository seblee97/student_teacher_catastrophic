from typing import Any
from typing import Dict

import constants
import numpy as np
from curricula import base_curriculum
from curricula import hard_steps_curriculum
from curricula import periodic_curriculum
from curricula import threshold_curriculum
from loggers import base_logger
from loggers import split_logger
from loggers import unified_logger
from ode import configuration
from ode import dynamics
from run import student_teacher_config
from utils import network_configuration


class ODERunner:
    """Runner for ode simulations."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        network_configuration: network_configuration.NetworkConfiguration,
    ) -> None:
        self._config = config
        self._network_configuration = network_configuration

        if config.consolidation_type is not None:
            if (
                config.consolidation_type
                == constants.Constants.NODE_CONSOLIDATION_HESSIAN
            ):
                self._importance = config.importance
            else:
                raise NotImplementedError(
                    "Only node-based consolidation with Hessian implemented in ODE"
                )
        else:
            self._importance = None

        self._curriculum = self._setup_curriculum()

        if self._config.implementation == constants.Constants.PYTHON:
            self._logger = self._setup_logger()

    def _setup_logger(self) -> base_logger.BaseLogger:
        if self._config.split_logging:
            logger = split_logger.SplitLogger(
                config=self._config,
                run_type=constants.Constants.ODE,
                network_config=self._network_configuration,
            )
        else:
            logger = unified_logger.UnifiedLogger(
                config=self._config,
                run_type=constants.Constants.ODE,
            )
        return logger

    def _setup_curriculum(self) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if self._config.stopping_condition == constants.Constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=self._config)
        elif self._config.stopping_condition == constants.Constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(config=self._config)
        elif self._config.stopping_condition == constants.Constants.SWITCH_STEPS:
            curriculum = hard_steps_curriculum.HardStepsCurriculum(config=self._config)
        else:
            raise ValueError(
                f"Stopping condition {self._config.stopping_condition} not recognised."
            )
        return curriculum

    def run(self):
        print("Beginning ODE solution...")
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

        # curriculum = (
        #     np.arange(0, self._config.total_training_steps, self._config.fixed_period)[
        #         1:
        #     ]
        #     / self._config.input_dimension
        # )

        time = self._config.total_training_steps / self._config.input_dimension

        ode = dynamics.StudentTeacherODE(
            overlap_configuration=ode_configuration,
            nonlinearity=self._config.student_nonlinearity,
            w_learning_rate=self._config.learning_rate,
            h_learning_rate=self._config.learning_rate,
            dt=timestep,
            soft_committee=self._config.soft_committee,
            train_first_layer=self._config.train_hidden_layers,
            train_head_layer=self._config.train_head_layer,
            frozen_feature=False,
            noise_stds=self._config.teacher_noises,
            copy_head_at_switch=self._config.copy_head_at_switch,
            importance=self._importance,
        )

        steps = 0
        task_steps = 0

        # step_increment = (timestep / time) * self._config.total_training_steps
        step_increment = timestep * self._config.input_dimension

        while ode.time < time:
            if steps % self._config.checkpoint_frequency == 0 and steps != 0:
                self._logger.checkpoint_df()

            self._logger.log_generalisation_errors(
                step=steps, generalisation_errors=[ode.error_1, ode.error_2]
            )
            self._logger.write_scalar_df(
                tag=constants.Constants.TEACHER_INDEX,
                step=steps,
                scalar=ode.active_teacher,
            )
            if self._config.log_overlaps and steps % self._config.log_frequency == 0:
                self._logger.log_network_configuration(
                    step=steps, network_config=ode.configuration
                )

            if self._curriculum.to_switch(
                task_step=task_steps, error=ode.current_teacher_error
            ):
                if self._config.log_overlaps:
                    self._logger.log_network_configuration(
                        step=steps, network_config=ode.configuration
                    )
                ode.switch_teacher()
                task_steps = 0

            ode.step(n_steps=step_increment)
            steps += step_increment
            task_steps += step_increment

        self._logger.checkpoint_df()

        # ode.save_to_csv(save_path=self._config.checkpoint_path)
        # ode.make_plot(
        #     save_path=self._config.checkpoint_path,
        #     total_time=self._config.total_training_steps,
        # )
