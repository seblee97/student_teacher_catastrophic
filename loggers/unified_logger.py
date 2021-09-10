import os
from typing import Dict

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
        self._df_columns: List = []
        self._logger_data: Dict = {}
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
        if tag not in self._logger_data:
            self._logger_data[tag] = {}
            self._df_columns.append(tag)

        self._logger_data[tag][step] = scalar

    def checkpoint_df(self) -> None:
        """Merge dataframe with previously saved checkpoint.

        Raises:
            AssertionError: if columns of dataframe to be appended do
            not match previous checkpoints.
        """
        assert (
            list(self._logger_data.keys()) == self._df_columns
        ), "Incorrect dataframe columns for merging"

        # only append header on first checkpoint/save.
        header = not os.path.exists(self._logfile_path)
        series_data = {k: pd.Series(v) for k, v in self._logger_data.items()}

        df = pd.DataFrame(series_data)

        for i in range(int(list(df.index)[0]), int(list(df.index)[-1])):
            if i not in list(df.index):
                df.loc[i] = pd.Series()
        df = df.sort_index()

        df.to_csv(
            self._logfile_path, mode="a", header=header, index=False
        )

        # reset logger in memory to empty.
        self._df_columns = []
        self._logger_data = {}
