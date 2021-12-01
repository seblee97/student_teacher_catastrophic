from typing import Any
from typing import Dict
from typing import Type

import numpy as np
from cata import constants
from cata.curricula import base_curriculum
from cata.curricula import hard_steps_curriculum
from cata.curricula import periodic_curriculum
from cata.curricula import threshold_curriculum
from cata.ode import configuration
from cata.ode import dynamics
from cata.run import student_teacher_config
from cata.utils import network_configuration
from run_modes import base_runner


class ODERunner(base_runner.BaseRunner):
    """Runner for ode simulations."""

    def __init__(
        self,
        config: Type[student_teacher_config.StudentTeacherConfig],
        network_configuration: network_configuration.NetworkConfiguration,
        unique_id: str = "",
    ) -> None:
        self._config = config
        self._network_configuration = network_configuration
        self._timestep = config.timestep
        self._num_teachers = 2

        if config.consolidation_type is not None:
            if config.consolidation_type == constants.NODE_CONSOLIDATION_HESSIAN:
                self._importance = config.importance
            else:
                raise NotImplementedError(
                    "Only node-based consolidation with Hessian implemented in ODE"
                )
        else:
            self._importance = None

        self._curriculum = self._setup_curriculum()
        self._ode = self._setup_ode()

        self._step_increment = self._timestep * config.input_dimension
        self._time = config.total_training_steps / config.input_dimension

        self._log_overlaps = config.log_overlaps
        self._log_frequency = config.log_frequency

        # if self._config.implementation == constants.PYTHON:
        #     self._logger = self._setup_logger()

        super().__init__(config=config, unique_id=unique_id)

    def _get_data_columns(self):
        columns = [constants.TEACHER_INDEX]
        generalisation_error_tags = [
            f"{constants.GENERALISATION_ERROR}_{i}" for i in range(self._num_teachers)
        ]
        log_generalisation_error_tags = [
            f"{constants.LOG_GENERALISATION_ERROR}_{i}"
            for i in range(self._num_teachers)
        ]
        columns.extend(generalisation_error_tags)
        columns.extend(log_generalisation_error_tags)
        if self._log_overlaps:
            sample_network_config = self._ode.configuration
            columns.extend(list(sample_network_config.dictionary.keys()))
        return columns

    def _setup_curriculum(self) -> base_curriculum.BaseCurriculum:
        """Initialise curriculum object (when to switch teacher,
        how to decide subsequent teacher etc.)

        Raises:
            ValueError: if stopping condition is not recognised.
        """
        if self._config.stopping_condition == constants.FIXED_PERIOD:
            curriculum = periodic_curriculum.PeriodicCurriculum(config=self._config)
        elif self._config.stopping_condition == constants.LOSS_THRESHOLDS:
            curriculum = threshold_curriculum.ThresholdCurriculum(config=self._config)
        elif self._config.stopping_condition == constants.SWITCH_STEPS:
            curriculum = hard_steps_curriculum.HardStepsCurriculum(config=self._config)
        else:
            raise ValueError(
                f"Stopping condition {self._config.stopping_condition} not recognised."
            )
        return curriculum

    def _setup_ode(self):
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

        ode = dynamics.StudentTeacherODE(
            overlap_configuration=ode_configuration,
            nonlinearity=self._config.student_nonlinearity,
            w_learning_rate=self._config.learning_rate,
            h_learning_rate=self._config.learning_rate,
            dt=self._timestep,
            soft_committee=self._config.soft_committee,
            train_first_layer=self._config.train_hidden_layers,
            train_head_layer=self._config.train_head_layer,
            frozen_feature=False,
            noise_stds=self._config.teacher_noises,
            copy_head_at_switch=self._config.copy_head_at_switch,
            importance=self._importance,
        )

        return ode

    def run(self):
        print("Beginning ODE solution...")
        if self._config.implementation == constants.CPP:
            self._run_cpp_ode()
        elif self._config.implementation == constants.PYTHON:
            self._run_python_ode()
        else:
            raise ValueError(
                f"Implementation type {self._config.implementation} not recognised."
            )

    def _run_cpp_ode(self):
        raise NotImplementedError

    def _run_python_ode(self):

        # curriculum = (
        #     np.arange(0, self._config.total_training_steps, self._config.fixed_period)[
        #         1:
        #     ]
        #     / self._config.input_dimension
        # )

        steps = 0
        task_steps = 0

        # step_increment = (timestep / time) * self._config.total_training_steps

        self._pretrain_log()

        while self._ode.time < self._time:

            steps += self._step_increment
            task_steps += self._step_increment

            step_logging_dict = self._ode_step(steps=steps)

            if steps % self._config.checkpoint_frequency == 0 and steps != 0:
                self._data_logger.checkpoint()

            if self._curriculum.to_switch(
                task_step=task_steps, error=self._ode.current_teacher_error
            ):
                if self._config.log_overlaps:
                    step_logging_dict = {
                        **step_logging_dict,
                        **self._ode.configuration.dictionary,
                    }
                self._ode.switch_teacher()
                task_steps = 0

            self._log_step_data(step=steps, logging_dict=step_logging_dict)

        self._data_logger.checkpoint()

        # ode.save_to_csv(save_path=self._config.checkpoint_path)
        # ode.make_plot(
        #     save_path=self._config.checkpoint_path,
        #     total_time=self._config.total_training_steps,
        # )]

    def _ode_step(self, steps: float):
        self._ode.step(n_steps=self._step_increment)

        step_logging_dict = {}

        step_logging_dict[f"{constants.GENERALISATION_ERROR}_0"] = self._ode.error_1
        step_logging_dict[f"{constants.GENERALISATION_ERROR}_1"] = self._ode.error_2
        step_logging_dict[f"{constants.LOG_GENERALISATION_ERROR}_0"] = np.log10(
            self._ode.error_1
        )
        step_logging_dict[f"{constants.LOG_GENERALISATION_ERROR}_1"] = np.log10(
            self._ode.error_2
        )
        step_logging_dict[constants.TEACHER_INDEX] = self._ode.active_teacher

        if self._log_overlaps and steps % self._log_frequency == 0:
            step_logging_dict = {
                **step_logging_dict,
                **self._ode.configuration.dictionary,
            }

        return step_logging_dict

    def _pretrain_log(self):
        pretrain_logging_dict = {}

        pretrain_logging_dict[f"{constants.GENERALISATION_ERROR}_0"] = self._ode.error_1
        pretrain_logging_dict[f"{constants.GENERALISATION_ERROR}_1"] = self._ode.error_2
        pretrain_logging_dict[f"{constants.LOG_GENERALISATION_ERROR}_0"] = np.log10(
            self._ode.error_1
        )
        pretrain_logging_dict[f"{constants.LOG_GENERALISATION_ERROR}_1"] = np.log10(
            self._ode.error_2
        )

        if self._log_overlaps:
            pretrain_logging_dict = {
                **pretrain_logging_dict,
                **self._ode.configuration.dictionary,
            }

        self._log_step_data(step=0, logging_dict=pretrain_logging_dict)

    def _log_step_data(self, step: int, logging_dict: Dict[str, Any]):
        for tag, scalar in logging_dict.items():
            self._data_logger.write_scalar(tag=tag, step=step, scalar=scalar)

    def post_process(self):
        self._plotter.load_data()
        self._plotter.plot_learning_curves()
