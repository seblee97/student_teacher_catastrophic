import os
from typing import List

import constants
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run import student_teacher_config
from utils import decorators
from utils import network_configuration


class Logger:
    """Class for logging experimental data."""

    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        csv_file_name: str,
    ):
        self._checkpoint_path = config.checkpoint_path
        self._logfile_path = os.path.join(self._checkpoint_path, csv_file_name)
        self._logger_df = pd.DataFrame()

    def write_scalar_df(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dataframe.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.

        Raises:
            AssertionError: if tag provided is not previously defined as a column.
        """
        self._logger_df.at[step, tag] = scalar

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
        network_configuration: network_configuration.NetworkConfiguration,
    ):
        for i, head in enumerate(network_configuration.student_head_weights):
            for j, weight in enumerate(head):
                self.write_scalar_df(
                    tag=f"{constants.STUDENT_HEAD}_{i}_{constants.WEIGHT}_{j}",
                    step=step,
                    scalar=weight,
                )
        for i, head in enumerate(network_configuration.teacher_head_weights):
            for j, weight in enumerate(head):
                self.write_scalar_df(
                    tag=f"{constants.TEACHER_HEAD}_{i}_{constants.WEIGHT}_{j}",
                    step=step,
                    scalar=weight,
                )
        for (i, j), overlap_value in np.ndenumerate(
            network_configuration.student_self_overlap
        ):
            self.write_scalar_df(
                tag=f"{constants.STUDENT_SELF}_{constants.OVERLAP}_{i}_{j}",
                step=step,
                scalar=overlap_value,
            )
        # for t, self_overlap in enumerate(network_configuration.teacher_self_overlaps):
        #     for (i, j), overlap_value in np.ndenumerate(self_overlap):
        #         self.write_scalar_df(
        #             tag=f"teacher_{t}_self_overlap_{i}_{j}",
        #             step=step,
        #             scalar=overlap_value,
        #         )
        for t, student_teacher_overlap in enumerate(
            network_configuration.student_teacher_overlaps
        ):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                self.write_scalar_df(
                    tag=f"{constants.STUDENT_TEACHER}_{t}_{constants.OVERLAP}_{i}_{j}",
                    step=step,
                    scalar=overlap_value,
                )

    @decorators.timer
    def checkpoint_df(self) -> None:
        """Merge dataframe with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        self._logger_df.to_csv(self._logfile_path, mode="a", header=header, index=False)

        # reset logger in memory to empty.
        self._logger_df = pd.DataFrame()
