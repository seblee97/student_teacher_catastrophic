from .base_logger import _BaseLogger

from typing import List, Union

import itertools

from utils import Parameters, visualise_matrix
from models.learners import _BaseLearner
from models.networks import _Teacher


class StudentTeacherLogger(_BaseLogger):

    """logging class for when teachers are initialised networks,
    regardless of input (standard student-teacher setup)
    """

    def __init__(self, config: Parameters):
        _BaseLogger.__init__(self, config)

    def _get_df_columns(self) -> List[str]:

        columns = [
            "training_loss",
            "mean_generalisation_error/log",
            "mean_generalisation_error/linear"
            ]

        columns.extend(self._get_training_metric_columns())
        columns.extend(self._get_student_head_weight_columns())
        columns.extend(self._get_student_teacher_hidden_overlap_columns())
        columns.extend(self._get_student_teacher_head_overlap_columns())
        columns.extend(self._get_student_hidden_self_overlap_columns())
        columns.extend(self._get_teacher_hidden_self_overlap_columns())
        columns.extend(self._get_student_head_self_overlap_columns())
        columns.extend(self._get_teacher_head_self_overlap_columns())
        columns.extend(self._get_teacher_teacher_hidden_overlap_columns())
        columns.extend(self._get_teacher_teacher_head_overlap_columns())

        return columns

    def _get_training_metric_columns(self) -> List[str]:
        columns = []
        # training metrics
        for t in range(self._num_teachers):
            columns.append("teacher_{}_generalisation_error/linear".format(t))
            columns.append("teacher_{}_generalisation_error/log".format(t))
            columns.append("teacher_{}_error_change/log".format(t))
            columns.append("teacher_{}_error_change/linear".format(t))
        return columns

    def _get_student_head_weight_columns(self) -> List[str]:
        columns = []
        # student head weights
        head_weight_combinations = itertools.product(*[
            range(self._num_teachers),
            range(self._student_hidden[-1] * self._output_dimension)
        ])

        for (t, output) in head_weight_combinations:
            columns.append(
                "student_head_{}_weight_{}".format(t, output)
                )
        return columns

    def _get_student_teacher_hidden_overlap_columns(self) -> List[str]:
        columns = []
        # student-teacher hidden overlaps
        for layer_index, layer in enumerate(self._student_hidden):
            hidden_overlap_combinations = \
                itertools.product(*[
                    range(self._num_teachers),
                    range(self._student_hidden[layer_index]),
                    range(self._teacher_hidden[layer_index])
                ])

            for (t, i, j) in hidden_overlap_combinations:
                columns.append(
                    "layer_{}_student_teacher_overlaps/{}/values_{}_{}".format(
                        layer_index, t, i, j
                    )
                )
        return columns

    def _get_student_teacher_head_overlap_columns(self) -> List[str]:
        columns = []
        # student-teacher head overlaps
        for h, latent in enumerate(self._student_hidden):
            output_head_overlap_combinations = \
                itertools.product(*[
                    range(self._num_teachers), range(self._num_teachers),
                    range(self._student_hidden[h]),
                    range(self._teacher_hidden[h])
                    ])

            for (t, t_rep, s_lat, t_lat) in output_head_overlap_combinations:
                columns.append(
                    "layer_output_head_{}_student_teacher_overlaps/"
                    "{}/values_{}_{}".format(t, t_rep, s_lat, t_lat)
                    )
        return columns

    def _get_student_hidden_self_overlap_columns(self) -> List[str]:
        columns = []
        # student hidden self-overlap
        for h, latent in enumerate(self._student_hidden):
            student_hidden_self_overlap_combos = \
                itertools.product(*[range(latent), range(latent)])
            for (i, j) in student_hidden_self_overlap_combos:
                columns.append(
                    "layer_{}_student_self_overlap/values_{}_{}".format(
                        h, i, j
                        )
                )
        return columns

    def _get_teacher_hidden_self_overlap_columns(self) -> List[str]:
        columns = []
        # teacher hidden self-overlap
        for t in range(self._num_teachers):
            for h, latent in enumerate(self._teacher_hidden):
                teacher_hidden_self_overlap_combos = \
                    itertools.product(*[range(latent), range(latent)])
                for (i, j) in teacher_hidden_self_overlap_combos:
                    columns.append(
                        "layer_{}_teacher_self_overlaps/"
                        "{}/values_{}_{}".format(
                            h, t, i, j
                            )
                    )
        return columns

    def _get_student_head_self_overlap_columns(self) -> List[str]:
        columns = []
        # student head self-overlap
        student_head_self_overlap_combos = itertools.product(*[
            range(self._num_teachers), range(self._output_dimension),
            range(self._output_dimension)
            ])

        for (t, d, d_rep) in student_head_self_overlap_combos:
            columns.append(
                "layer_output_head_{}_student_self_overlap/"
                "values_{}_{}".format(t, d, d_rep)
            )
        return columns

    def _get_teacher_head_self_overlap_columns(self) -> List[str]:
        columns = []
        # teacher head self-overlap
        teacher_head_self_overlap_combos = itertools.product(*[
            range(self._num_teachers), range(self._num_teachers),
            range(self._output_dimension), range(self._output_dimension)
            ])

        for (t, t_rep, d, d) in teacher_head_self_overlap_combos:
            columns.append(
                "layer_output_head_{}_teacher_self_overlaps/{}/"
                "values_{}_{}".format(t, t_rep, d, d)
            )
        return columns

    def _get_teacher_teacher_hidden_overlap_columns(self) -> List[str]:
        columns = []
        teacher_pairs = itertools.combinations(range(self._num_teachers), 2)
        # teacher - teacher hidden overlaps
        for h, hidden in enumerate(self._teacher_hidden):
            for (t1, t2) in teacher_pairs:
                index_combos = itertools.product(
                    *[range(hidden), range(hidden)]
                    )
                for (i, j) in index_combos:
                    columns.append(
                        "layer_{}_teacher_teacher_overlaps/"
                        "{}_{}/values_{}_{}".format(
                            h, t1, t2, i, j
                            )
                        )
        return columns

    def _get_teacher_teacher_head_overlap_columns(self) -> List[str]:
        columns = []
        # teacher - teacher head overlaps
        for t in range(self._num_teachers):
            for t1 in range(self._num_teachers):
                for t2 in range(t1 + 1, self._num_teachers):
                    index_combos = itertools.product(*[
                        range(self._teacher_hidden[-1]),
                        range(self._output_dimension)
                        ])
                    for (i, j) in index_combos:
                        columns.append(
                            "layer_output_head_{}"
                            "_teacher_teacher_overlaps/"
                            "{}_{}/values_{}_{}".format(
                                t, t1, t2, i, j
                                )
                            )
        return columns

    def _compute_layer_overlaps(
        self,
        layer: str,
        student_network: _BaseLearner,
        teacher_networks: List[_Teacher],
        head: Union[None, int],
        step_count: int
    ) -> None:
        """
        computes overlap of given layer of student_network and
        teacher networks.

        :param layer: index of layer to compute overlap matrices for
        :param student_network: student_network_module
        :param teacher_networks: list of teacher network modules
        :param step_count: current step of training
        """
        # extract layer weights
        if head is None:
            student_layer = \
                student_network.state_dict()[
                    'layers.{}.weight'.format(layer)
                    ].data
            teacher_layers = [
                teacher.state_dict()['layers.{}.weight'.format(layer)].data
                for teacher in teacher_networks
                ]
        else:
            student_layer = \
                student_network.state_dict()[
                    'heads.{}.weight'.format(str(head))
                    ].data
            teacher_layers = [
                teacher.state_dict()['output_layer.weight'].data
                for teacher in teacher_networks
                ]
            layer = layer + "_head_{}".format(str(head))

        # compute overlap matrices
        student_self_overlap = (
            student_layer.mm(student_layer.t()) / self._input_dimension
            ).cpu().numpy()
        if head is None:
            student_teacher_overlaps = [
                (
                    student_layer.mm(teacher_layer.t()) / self._input_dimension
                ).cpu().numpy()
                for teacher_layer in teacher_layers
                ]
        else:
            student_teacher_overlaps = [
                (
                    student_layer.t().mm(teacher_layer) / self._input_dimension
                ).cpu().numpy()
                for teacher_layer in teacher_layers
                ]
        teacher_self_overlaps = [
                (
                    teacher_layer.mm(teacher_layer.t()) / self._input_dimension
                ).cpu().numpy()
                for teacher_layer in teacher_layers
                ]
        teacher_pairs = list(
            itertools.combinations(range(len(teacher_layers)), 2)
            )
        teacher_teacher_overlaps = {(i, j): (
            teacher_layers[i].mm(teacher_layers[j].t()) / self._input_dimension
            ).cpu().numpy() for (i, j) in teacher_pairs
            }

        # log overlap values (scalars vs image graphs below)
        def log_matrix_values(log_name: str, matrix):
            matrix_shape = matrix.shape
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    if self._verbose_tb > 1:
                        self._writer.add_scalar(
                            "layer_{}_{}/values_{}_{}".format(
                                layer, log_name, i, j
                                ),
                            matrix[i][j],
                            step_count
                            )
                    if self._log_to_df:
                        self._logger_df.at[
                            step_count,
                            "layer_{}_{}/values_{}_{}".format(
                                layer, log_name, i, j
                                )
                            ] = matrix[i][j]

        log_matrix_values("student_self_overlap", student_self_overlap)
        for s, student_teacher_overlap in enumerate(student_teacher_overlaps):
            log_matrix_values(
                "student_teacher_overlaps/{}".format(s),
                student_teacher_overlap
                )
        for s, teacher_self_overlap in enumerate(teacher_self_overlaps):
            log_matrix_values(
                "teacher_self_overlaps/{}".format(s), teacher_self_overlap
                )
        for (i, j) in teacher_teacher_overlaps:
            log_matrix_values(
                "teacher_teacher_overlaps/{}_{}".format(i, j),
                teacher_teacher_overlaps[(i, j)]
                )

        # generate visualisations
        if self._verbose_tb > 1:
            student_self_fig = visualise_matrix(
                student_self_overlap, fig_title=r"$Q_{ik}^\mu$"
                )
            teacher_cross_figs = {(i, j): visualise_matrix(
                teacher_teacher_overlaps[(i, j)], fig_title=r"$T_{nm}$"
                ) for (i, j) in teacher_teacher_overlaps
                }
            teacher_self_figs = [
                visualise_matrix(teacher_self_overlap, fig_title=r"$T_{nm}$")
                for teacher_self_overlap in teacher_self_overlaps
            ]
            student_teacher_figs = [
                visualise_matrix(
                    student_teacher_overlap, fig_title=r"$R_{in}^\mu$"
                    )
                for t, student_teacher_overlap
                in enumerate(student_teacher_overlaps)
            ]

        # log visualisations
        if self._verbose_tb > 1:
            self._writer.add_figure(
                "layer_{}_student_self_overlap".format(str(layer)),
                student_self_fig, step_count
                )
            for t, student_teacher_fig in enumerate(student_teacher_figs):
                self._writer.add_figure(
                    "layer_{}_student_teacher_overlaps/teacher_{}".format(
                        layer, t
                        ),
                    student_teacher_fig, step_count
                    )
            for t, teacher_self_fig in enumerate(teacher_self_figs):
                self._writer.add_figure(
                    "layer_{}_teacher_self_overlaps/teacher_{}".format(
                        layer, t
                        ),
                    teacher_self_fig, step_count
                    )
            for (i, j), teacher_cross_fig in list(teacher_cross_figs.items()):
                self._writer.add_figure(
                    "layer_{}_teacher_cross_overlaps/teacher{}x{}".format(
                        layer, i, j
                        ),
                    teacher_cross_fig, step_count
                    )
