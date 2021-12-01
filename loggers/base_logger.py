import abc
import os
from typing import List

import constants
import numpy as np
from run import student_teacher_config
from utils import decorators
from utils import network_configuration


class BaseLogger(abc.ABC):
    """Base class for logging experimental data."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        run_type: str,
    ):
        self._checkpoint_path = config.checkpoint_path
        self._num_teachers = config.num_teachers
        self._run_type = run_type

        if self._run_type == constants.SIM:
            self._csv_file_name = constants.NETWORK_CSV
        elif self._run_type == constants.ODE:
            self._csv_file_name = constants.ODE_CSV

        self._setup_loggers()

    @abc.abstractmethod
    def _setup_loggers(self):
        """Initialise relevant logging dataframes."""
        pass

    @abc.abstractmethod
    def write_scalar_df(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dataframe.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.

        Raises:
            AssertionError: if tag provided is not previously defined as a column.
        """
        pass

    def log_generalisation_errors(self, step: int, generalisation_errors: List[float]):
        for i, error in enumerate(generalisation_errors):
            self.write_scalar_df(
                tag=f"{constants.GENERALISATION_ERROR}_{i}",
                step=step,
                scalar=error,
            )
            self.write_scalar_df(
                tag=f"{constants.LOG_GENERALISATION_ERROR}_{i}",
                step=step,
                scalar=np.log10(error),
            )

    def log_network_configuration(
        self,
        step: int,
        network_config: network_configuration.NetworkConfiguration,
    ):
        for i, head in enumerate(network_config.student_head_weights):
            for j, weight in enumerate(head):
                self.write_scalar_df(
                    tag=f"{constants.STUDENT_HEAD}_{i}_{constants.WEIGHT}_{j}",
                    step=step,
                    scalar=weight,
                )
        for i, head in enumerate(network_config.teacher_head_weights):
            for j, weight in enumerate(head):
                self.write_scalar_df(
                    tag=f"{constants.TEACHER_HEAD}_{i}_{constants.WEIGHT}_{j}",
                    step=step,
                    scalar=weight,
                )
        for (i, j), overlap_value in np.ndenumerate(
            network_config.student_self_overlap
        ):
            self.write_scalar_df(
                tag=f"{constants.STUDENT_SELF}_{constants.OVERLAP}_{i}_{j}",
                step=step,
                scalar=overlap_value,
            )
        for t, student_teacher_overlap in enumerate(
            network_config.student_teacher_overlaps
        ):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                self.write_scalar_df(
                    tag=f"{constants.STUDENT_TEACHER}_{t}_{constants.OVERLAP}_{i}_{j}",
                    step=step,
                    scalar=overlap_value,
                )
        # if using node consolidation
        for (i, j), overlap_value in np.ndenumerate(
            network_config.student_old_student_overlap
        ):
            self.write_scalar_df(
                tag=f"{constants.STUDENT_OLD_STUDENT}_{constants.OVERLAP}_{i}_{j}",
                step=step,
                scalar=overlap_value,
            )

    @abc.abstractmethod
    @decorators.timer
    def checkpoint_df(self) -> None:
        """Merge dataframe with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        pass
