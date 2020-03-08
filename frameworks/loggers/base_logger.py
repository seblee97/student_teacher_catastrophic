from abc import ABC, abstractmethod

from typing import Dict, List

import logging

import os

from tensorboardX import SummaryWriter

import pandas as pd
import torch

import time

class BaseLogger(ABC):

    def __init__(self, config: Dict):

        self._extract_parameters(config=config)

        # initialise general tensorboard writer
        self._writer = SummaryWriter(self.checkpoint_path)
        # initialise separate tb writers for each teacher
        self._teacher_writers = [SummaryWriter("{}/teacher_{}".format(self.checkpoint_path, t)) for t in range(self.num_teachers)]

        # initialise dataframe to log metrics
        if self.log_to_df:
            self._logger_df = pd.DataFrame()

        # initialise logger
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # create handlers
        # std_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('{}/experiment.log'.format(self.checkpoint_path))

        # Add handlers to the logger
        # _.addHandler(std_handler)
        self._logger.addHandler(file_handler)

    def _extract_parameters(self, config: Dict):
        self.checkpoint_path = config.get("checkpoint_path")
        self.log_to_df = config.get("log_to_df")
        self.logfile_path = config.get("logfile_path")
        self.verbose_tb = config.get("verbose_tb")
        self.checkpoint_frequency = config.get("checkpoint_frequency")

        self.student_hidden = config.get(["model", "student_hidden_layers"])
        self.num_teachers = config.get(["task", "num_teachers"])
        self.input_dimension = config.get(["model", "input_dimension"])

    def log(self, statement: str):
        self._logger.info(statement)

    def add_row(self, row_label):
        self._logger_df.append(pd.Series(name=row_label))

    def write_scalar_tb(self, tag: str, scalar: float, step: int, teacher_index=None):
        if teacher_index:
            self._teacher_writers[teacher_index].add_scalar(tag, scalar, step)
        else:
            self._writer.add_scalar(tag, scalar, step)

    def write_scalar_df(self, tag: str, scalar: float, step: int):
        self._logger_df.at[step, tag] = scalar

    def _log_output_weights(self, step_count: int, student_network) -> None:
        """
        extract and log output weights of student.
        """
        # extract output layer weights for student
        student_output_layer_weights = student_network._get_head_weights()
        for h, head in enumerate(student_output_layer_weights):
            flattened_weights = torch.flatten(head)
            for w, weight in enumerate(flattened_weights):
                if self.verbose_tb:
                    self._writer.add_scalar('student_head_{}_weight_{}'.format(h, w), float(weight), step_count)
                if self.log_to_df:
                    self._logger_df.at[step_count, 'student_head_{}_weight_{}'.format(h, w)] = float(weight)

    def _compute_overlap_matrices(self, student_network, teacher_networks, step_count: int) -> None:
        for layer in range(len(self.student_hidden)):
            self._compute_layer_overlaps(layer=str(layer), student_network=student_network, teacher_networks=teacher_networks, head=None, step_count=step_count)

        for head_index in range(self.num_teachers):
            self._compute_layer_overlaps(layer="output", student_network=student_network, teacher_networks=teacher_networks, head=head_index, step_count=step_count)

    @abstractmethod
    def _compute_layer_overlaps(self, layer: int, student_network, teacher_networks: List, head: int, step_count: int):
        raise NotImplementedError("Base class method")

    def checkpoint_df(self, step: int):
        self._logger.info("Checkpointing Dataframe...")
        t0 = time.time()

        # check for existing dataframe
        if step > self.checkpoint_frequency:
            repeated_index = step - self.checkpoint_frequency
            previous_df = pd.read_csv(self.logfile_path, index_col=0)
            merged_df = pd.concat([previous_df, self._logger_df])
            merged_row = merged_df.loc[repeated_index].groupby(level=0).max()
            merged_df.drop([repeated_index])
            merged_df = pd.concat([merged_df, merged_row]).sort_index()
        else:
            merged_df = self._logger_df

        merged_df.to_csv(self.logfile_path)
        if self.log_to_df:
            self._logger_df = pd.DataFrame()
        self._logger.info("Dataframe checkpointed in {}s".format(round(time.time() - t0, 5)))

    def _consolidate_dfs(self):
        self._logger.info("Consolidating/Merging all dataframes..")
        import pdb; pdb.set_trace()
        all_df_paths = [os.path.join(self.checkpoint_path, f) for f in os.listdir(self.checkpoint_path) if f.endswith('.csv')]
        all_dfs = [pd.read_csv(df_path) for df_path in all_df_paths]
        merged_df = pd.concat(all_dfs)
        merged_df.to_csv("{}.csv".format(self.logfile_path))

        # remove indiviudal dataframes
        for df in all_df_paths:
            os.remove(df)
