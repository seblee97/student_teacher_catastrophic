from abc import ABC, abstractmethod

from typing import Dict, List

import logging
import os
import time

from tensorboardX import SummaryWriter

import pandas as pd
import torch
import torch.nn as nn

class _BaseLogger(ABC):

    def __init__(self, config: Dict):

        self._extract_parameters(config=config)

        # initialise general tensorboard writer
        self._writer = SummaryWriter(self._checkpoint_path)
        
        # initialise separate tb writers for each teacher
        self._teacher_writers = [SummaryWriter("{}/teacher_{}".format(self._checkpoint_path, t)) for t in range(self._num_teachers)]

        # initialise dataframe to log metrics
        if self._log_to_df:
            self._logger_df = pd.DataFrame()

        # initialise logger
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        file_handler = logging.FileHandler('{}/experiment.log'.format(self._checkpoint_path))

        # Add handlers to the logger
        self._logger.addHandler(file_handler)

    def _extract_parameters(self, config: Dict) -> None:
        """get relevant parameters from config and make attributes of class"""
        self._checkpoint_path = config.get("checkpoint_path")
        self._log_to_df = config.get("log_to_df")
        self._logfile_path = config.get("logfile_path")
        self._verbose_tb = config.get("verbose_tb")
        self._checkpoint_frequency = config.get("checkpoint_frequency")

        self._student_hidden = config.get(["model", "student_hidden_layers"])
        self._num_teachers = config.get(["task", "num_teachers"])
        self._input_dimension = config.get(["model", "input_dimension"])

    def log(self, statement: str) -> None:
        """add to log file"""
        self._logger.info(statement)

    def add_row(self, row_label) -> None:
        """add row to logging dataframe"""
        self._logger_df.append(pd.Series(name=row_label))

    def write_scalar_tb(self, tag: str, scalar: float, step: int, teacher_index: int=None) -> None:
        """
        write (scalar) data to tensorboard.

        :param tag: tag for data to be logged
        :param scalar: data to be written
        :param step: current step count (x axis on tb)
        :param teacher_index: index of writer (uses general writer if None)
        """
        if teacher_index is not None:
            self._teacher_writers[teacher_index].add_scalar(tag, scalar, step)
        else:
            self._writer.add_scalar(tag, scalar, step)

    def write_scalar_df(self, tag: str, scalar: float, step: int) -> None:
        """
        write (scalar) data to dataframe.

        :param tag: tag for data to be logged
        :param scalar: data to be written
        :param step: current step count (x axis on tb)
        """
        self._logger_df.at[step, tag] = scalar

    def _log_output_weights(self, step_count: int, student_network: nn.Module) -> None:
        """
        extract and log output weights of student.

        :param step_count: current step of training
        :param student_network: student network module
        """
        # extract output layer weights for student
        student_output_layer_weights = student_network._get_head_weights()
        for h, head in enumerate(student_output_layer_weights):
            flattened_weights = torch.flatten(head)
            for w, weight in enumerate(flattened_weights):
                if self._verbose_tb:
                    self._writer.add_scalar('student_head_{}_weight_{}'.format(h, w), float(weight), step_count)
                if self._log_to_df:
                    self._logger_df.at[step_count, 'student_head_{}_weight_{}'.format(h, w)] = float(weight)

    def _compute_overlap_matrices(self, student_network: nn.Module, teacher_networks: List[nn.Module], step_count: int) -> None:
        """
        calculate overlap matrices between student and teacher and across student/teacher layers

        :param student_network: student_network_module
        :param teacher_networks: list of teacher network modules
        :param step_count: current step of training
        """
        for layer in range(len(self._student_hidden)):
            self._compute_layer_overlaps(layer=str(layer), student_network=student_network, teacher_networks=teacher_networks, head=None, step_count=step_count)

        for head_index in range(self._num_teachers):
            self._compute_layer_overlaps(layer="output", student_network=student_network, teacher_networks=teacher_networks, head=head_index, step_count=step_count)

    @abstractmethod
    def _compute_layer_overlaps(self, layer: int, student_network: nn.Module, teacher_networks: List[nn.Module], head: int, step_count: int) -> None:
        """
        computes overlap of given layer of student_network and teacher networks.

        :param layer: index of layer to compute overlap matrices for
        :param student_network: student_network_module
        :param teacher_networks: list of teacher network modules
        :param step_count: current step of training
        """
        raise NotImplementedError("Base class method")

    def checkpoint_df(self, step: int) -> None:
        """save dataframe"""
        self._logger.info("Checkpointing Dataframe...")
        t0 = time.time()

        # check for existing dataframe
        if step > self._checkpoint_frequency:
            repeated_index = step - self._checkpoint_frequency
            previous_df = pd.read_csv(self._logfile_path, index_col=0)
            merged_df = pd.concat([previous_df, self._logger_df])
            merged_row = merged_df.loc[repeated_index].groupby(level=0).max()
            merged_df.drop([repeated_index])
            merged_df = pd.concat([merged_df, merged_row]).sort_index()
        else:
            merged_df = self._logger_df

        merged_df.to_csv(self._logfile_path)
        if self._log_to_df:
            self._logger_df = pd.DataFrame()
        self._logger.info("Dataframe checkpointed in {}s".format(round(time.time() - t0, 5)))

    def _consolidate_dfs(self) -> None:
        self._logger.info("Consolidating/Merging all dataframes..")
        import pdb; pdb.set_trace()
        all_df_paths = [os.path.join(self._checkpoint_path, f) for f in os.listdir(self._checkpoint_path) if f.endswith('.csv')]
        all_dfs = [pd.read_csv(df_path) for df_path in all_df_paths]
        merged_df = pd.concat(all_dfs)
        merged_df.to_csv("{}.csv".format(self._logfile_path))

        # remove indiviudal dataframes
        for df in all_df_paths:
            os.remove(df)
