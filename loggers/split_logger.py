import os
from typing import Dict

import numpy as np
import pandas as pd

import constants
from loggers import base_logger
from run import student_teacher_config
from utils import network_configuration


class SplitLogger(base_logger.BaseLogger):
    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        run_type: str,
        network_config: network_configuration.NetworkConfiguration,
    ):
        self._loggers: Dict[str, pd.DataFrame] = {}
        self._logfile_paths: Dict[str, str] = {}
        self._initialisation_network_config = network_config

        super().__init__(config=config, run_type=run_type)

    def _setup_loggers(self):
        """Initialise relevant logging dataframes. Here, one dataframe per tag."""
        os.makedirs(os.path.join(self._checkpoint_path, self._run_type), exist_ok=True)
        self._setup_error_loggers()
        self._setup_overlap_loggers()

    def _setup_error_loggers(self):
        for i in range(self._num_teachers):
            tag = f"{constants.Constants.GENERALISATION_ERROR}_{i}"
            self._loggers[tag] = pd.DataFrame()
            self._logfile_paths[tag] = os.path.join(
                self._checkpoint_path, self._run_type, f"{tag}_{self._csv_file_name}"
            )
            log_tag = f"{constants.Constants.LOG_GENERALISATION_ERROR}_{i}"
            self._loggers[log_tag] = pd.DataFrame()
            self._logfile_paths[log_tag] = os.path.join(
                self._checkpoint_path,
                self._run_type,
                f"{log_tag}_{self._csv_file_name}",
            )

    def _setup_overlap_loggers(self):
        for i, head in enumerate(
            self._initialisation_network_config.student_head_weights
        ):
            for j, head in enumerate(head):
                tag = f"{constants.Constants.STUDENT_HEAD}_{i}_{constants.Constants.WEIGHT}_{j}"
                self._loggers[tag] = pd.DataFrame()
                self._logfile_paths[tag] = os.path.join(
                    self._checkpoint_path,
                    self._run_type,
                    f"{tag}_{self._csv_file_name}",
                )
        for i, head in enumerate(
            self._initialisation_network_config.teacher_head_weights
        ):
            for j, weight in enumerate(head):
                tag = f"{constants.Constants.TEACHER_HEAD}_{i}_{constants.Constants.WEIGHT}_{j}"
                self._loggers[tag] = pd.DataFrame()
                self._logfile_paths[tag] = os.path.join(
                    self._checkpoint_path,
                    self._run_type,
                    f"{tag}_{self._csv_file_name}",
                )
        for (i, j), overlap_value in np.ndenumerate(
            self._initialisation_network_config.student_self_overlap
        ):
            tag = f"{constants.Constants.STUDENT_SELF}_{constants.Constants.OVERLAP}_{i}_{j}"
            self._loggers[tag] = pd.DataFrame()
            self._logfile_paths[tag] = os.path.join(
                self._checkpoint_path, self._run_type, f"{tag}_{self._csv_file_name}"
            )
        for t, student_teacher_overlap in enumerate(
            self._initialisation_network_config.student_teacher_overlaps
        ):
            for (i, j), overlap_value in np.ndenumerate(student_teacher_overlap):
                tag = f"{constants.Constants.STUDENT_TEACHER}_{t}_{constants.Constants.OVERLAP}_{i}_{j}"
                self._loggers[tag] = pd.DataFrame()
                self._logfile_paths[tag] = os.path.join(
                    self._checkpoint_path,
                    self._run_type,
                    f"{tag}_{self._csv_file_name}",
                )

    def write_scalar_df(self, tag: str, step: int, scalar: float) -> None:
        """Write (scalar) data to dataframe.

        Args:
            tag: tag for data to be logged.
            step: current step count.
            scalar: data to be written.

        Raises:
            AssertionError: if tag provided is not previously defined as a column.
        """
        self._loggers[tag].at[step, tag] = scalar

    def checkpoint_df(self) -> None:
        """Merge dataframe with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        for tag, df_path in self._logfile_paths.items():
            # only append header on first checkpoint/save.
            header = not os.path.exists(df_path)
            self._loggers[tag].to_csv(df_path, mode="a", header=header, index=False)

            # reset loggers in memory to empty.
            self._loggers[tag] = pd.DataFrame()
