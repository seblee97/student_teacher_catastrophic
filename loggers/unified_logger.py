import os

import pandas as pd

from loggers import base_logger
from run import student_teacher_config


class UnifiedLogger(base_logger.BaseLogger):
    def __init__(
        self,
        config: student_teacher_config.StudentTeacherConfiguration,
        run_type: str,
    ):
        self._logger_df: pd.DataFrame
        self._logfile_path: str

        super().__init__(config=config, run_type=run_type)

    def _setup_loggers(self):
        """Initialise relevant logging dataframes. Here, single dataframe."""
        self._logger_df = pd.DataFrame()
        self._logfile_path = os.path.join(self._checkpoint_path, self._csv_file_name)

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
