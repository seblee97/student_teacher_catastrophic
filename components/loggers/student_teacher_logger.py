from .base_logger import _BaseLogger

from typing import Dict, List, Union

import itertools
import torch.nn as nn

from utils import Parameters, visualise_matrix
from models.learners import _BaseLearner
from models.networks import _Teacher

class StudentTeacherLogger(_BaseLogger):

    """logging class for when teachers are initialised networks, regardless of input (standard student-teacher setup)"""

    def __init__(self, config: Parameters):
        _BaseLogger.__init__(self, config)

    def _compute_layer_overlaps(self, layer: str, student_network: _BaseLearner, teacher_networks: List[_Teacher], head: Union[None, int], step_count: int) -> None:
        """
        computes overlap of given layer of student_network and teacher networks.

        :param layer: index of layer to compute overlap matrices for
        :param student_network: student_network_module
        :param teacher_networks: list of teacher network modules
        :param step_count: current step of training
        """
        # extract layer weights
        if head is None:
            student_layer = student_network.state_dict()['layers.{}.weight'.format(layer)].data
            teacher_layers = [teacher.state_dict()['layers.{}.weight'.format(layer)].data for teacher in teacher_networks]
        else:
            student_layer = student_network.state_dict()['heads.{}.weight'.format(str(head))].data
            teacher_layers = [teacher.state_dict()['output_layer.weight'].data for teacher in teacher_networks]
            layer = layer + "_head_{}".format(str(head))

        # compute overlap matrices
        student_self_overlap = (student_layer.mm(student_layer.t()) / self._input_dimension).cpu().numpy()
        if head is None:
            student_teacher_overlaps = [(student_layer.mm(teacher_layer.t()) / self._input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        else:
            student_teacher_overlaps = [(student_layer.t().mm(teacher_layer) / self._input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        teacher_self_overlaps = [(teacher_layer.mm(teacher_layer.t()) / self._input_dimension).cpu().numpy() for teacher_layer in teacher_layers]
        teacher_pairs = list(itertools.combinations(range(len(teacher_layers)), 2))
        teacher_teacher_overlaps = {(i, j): (teacher_layers[i].mm(teacher_layers[j].t()) / self._input_dimension).cpu().numpy() for (i, j) in teacher_pairs}

        # log overlap values (scalars vs image graphs below)
        def log_matrix_values(log_name: str, matrix):
            matrix_shape = matrix.shape
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    if self._verbose_tb:
                        self._writer.add_scalar("layer_{}_{}/values_{}_{}".format(layer, log_name, i, j), matrix[i][j], step_count)
                    if self._log_to_df:
                        self._logger_df.at[step_count, "layer_{}_{}/values_{}_{}".format(layer, log_name, i, j)] = matrix[i][j]
        
        log_matrix_values("student_self_overlap", student_self_overlap)
        for s, student_teacher_overlap in enumerate(student_teacher_overlaps):
            log_matrix_values("student_teacher_overlaps/{}".format(s), student_teacher_overlap)
        for s, teacher_self_overlap in enumerate(teacher_self_overlaps):
            log_matrix_values("teacher_self_overlaps/{}".format(s), teacher_self_overlap)
        for (i, j) in teacher_teacher_overlaps:
            log_matrix_values("teacher_teacher_overlaps/{}_{}".format(i, j), teacher_teacher_overlaps[(i, j)])

        # generate visualisations
        if self._verbose_tb:
            student_self_fig = visualise_matrix(student_self_overlap, fig_title=r"$Q_{ik}^\mu$")
            teacher_cross_figs = {(i, j):
                visualise_matrix(teacher_teacher_overlaps[(i, j)], fig_title=r"$T_{nm}$") \
                for (i, j) in teacher_teacher_overlaps
            }
            teacher_self_figs = [
                visualise_matrix(teacher_self_overlap, fig_title=r"$T_{nm}$") \
                for teacher_self_overlap in teacher_self_overlaps
            ]
            student_teacher_figs = [
                visualise_matrix(student_teacher_overlap, fig_title=r"$R_{in}^\mu$") \
                for t, student_teacher_overlap in enumerate(student_teacher_overlaps)
            ]

        # log visualisations
        if self._verbose_tb:
            self._writer.add_figure("layer_{}_student_self_overlap".format(str(layer)), student_self_fig, step_count)
            for t, student_teacher_fig in enumerate(student_teacher_figs):
                self._writer.add_figure("layer_{}_student_teacher_overlaps/teacher_{}".format(layer, t), student_teacher_fig, step_count)
            for t, teacher_self_fig in enumerate(teacher_self_figs):
                self._writer.add_figure("layer_{}_teacher_self_overlaps/teacher_{}".format(layer, t), teacher_self_fig, step_count)
            for (i, j), teacher_cross_fig in list(teacher_cross_figs.items()):
                self._writer.add_figure("layer_{}_teacher_cross_overlaps/teacher{}x{}".format(layer, i, j), teacher_cross_fig, step_count)