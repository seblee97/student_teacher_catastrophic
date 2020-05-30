from .base_logger import _BaseLogger

from typing import List, Union, Tuple, Dict

import itertools

import torch
import numpy as np

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
        columns.extend(self._get_student_teacher_hidden_overlap_diff_columns())
        columns.extend(self._get_student_teacher_head_overlap_columns())
        columns.extend(self._get_student_hidden_self_overlap_columns())
        columns.extend(self._get_student_hidden_grad_overlap_columns())
        columns.extend(self._get_teacher_hidden_self_overlap_columns())
        columns.extend(self._get_student_head_self_overlap_columns())
        columns.extend(self._get_student_head_grad_overlap_columns())
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

    def _get_student_teacher_hidden_overlap_diff_columns(self) -> List[str]:
        columns = []
        for layer_index, layer in enumerate(self._student_hidden):
            hidden_overlap_combinations = \
                itertools.product(*[
                    range(self._num_teachers),
                    range(self._student_hidden[layer_index]),
                    range(self._teacher_hidden[layer_index]),
                    range(self._student_hidden[layer_index]),
                    range(self._teacher_hidden[layer_index])
                ])

            for (t, i1, j1, i2, j2) in hidden_overlap_combinations:
                columns.append(
                    (
                        "layer_{}_student_teacher_overlaps/{}/"
                        "difference_{}_{}_vs_{}_{}"
                    ).format(
                        layer_index, t, i1, j1, i2, j2
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

    def _get_student_hidden_grad_overlap_columns(self) -> List[str]:
        columns = []
        for h, latent in enumerate(self._student_hidden):
            student_hidden_grad_overlap_combos = \
                itertools.product(*[range(latent), range(latent)])
            for (i, j) in student_hidden_grad_overlap_combos:
                columns.append(
                    "layer_{}_student_grad_student_overlap"
                    "/values_{}_{}".format(
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

    def _get_student_head_grad_overlap_columns(self) -> List[str]:
        columns = []
        # student head self-overlap
        student_head_grad_overlap_combos = itertools.product(*[
            range(self._num_teachers), range(self._output_dimension),
            range(self._output_dimension)
            ])

        for (t, d, d_rep) in student_head_grad_overlap_combos:
            columns.append(
                "layer_output_head_{}_student_grad_student_overlap/"
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
        layer, student_layer, student_layer_gradients, teacher_layers = \
            self._extract_layer_weights(
                layer=layer, student_network=student_network,
                teacher_networks=teacher_networks, head=head
            )

        student_self_overlap = self._compute_student_self_overlap(
            student_layer=student_layer
            )
        self._log_matrix_values(
            layer=layer, step_count=step_count,
            log_name="student_self_overlap",
            matrix=student_self_overlap
            )

        if student_layer_gradients is not None:
            student_grad_student_overlap = \
                self._compute_student_grad_student_overlap(
                    student_layer=student_layer,
                    student_layer_gradients=student_layer_gradients
                )
            self._log_matrix_values(
                layer=layer, step_count=step_count,
                log_name="student_grad_student_overlap",
                matrix=student_grad_student_overlap
            )

        student_teacher_overlaps = self._compute_student_teacher_overlaps(
            student_layer=student_layer, teacher_layers=teacher_layers,
            head=head
        )

        for s, student_teacher_overlap in enumerate(student_teacher_overlaps):
            self._log_matrix_values(
                layer=layer, step_count=step_count,
                log_name="student_teacher_overlaps/{}".format(s),
                matrix=student_teacher_overlap
                )
            if head is None:
                student_teacher_overlap_differences = \
                    self._compute_overlap_differences(
                        student_teacher_overlap
                        )
                self._log_difference_matrix_values(
                    layer=layer, step_count=step_count,
                    log_name="student_teacher_overlaps/{}".format(s),
                    matrix=student_teacher_overlap_differences
                )

        teacher_self_overlaps = self._compute_teacher_self_overlaps(
            teacher_layers=teacher_layers
        )
        for s, teacher_self_overlap in enumerate(teacher_self_overlaps):
            self._log_matrix_values(
                layer=layer, step_count=step_count,
                log_name="teacher_self_overlaps/{}".format(s),
                matrix=teacher_self_overlap
                )

        teacher_teacher_overlaps = self._compute_teacher_teacher_overlaps(
            teacher_layers=teacher_layers
        )
        for (i, j) in teacher_teacher_overlaps:
            self._log_matrix_values(
                layer=layer, step_count=step_count,
                log_name="teacher_teacher_overlaps/{}_{}".format(i, j),
                matrix=teacher_teacher_overlaps[(i, j)]
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

    def _extract_layer_weights(
        self,
        layer: str,
        student_network: _BaseLearner,
        teacher_networks: List[_Teacher],
        head: Union[None, int],
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # extract layer weights
        if head is None:
            student_layer = \
                student_network.state_dict()[
                    'layers.{}.weight'.format(layer)
                    ].data
            student_layer_gradients = \
                student_network.layers[int(layer)].weight.grad
            teacher_layers = [
                teacher.state_dict()['layers.{}.weight'.format(layer)].data
                for teacher in teacher_networks
                ]
        else:
            if self._learner_configuration == "meta":
                student_layer = \
                    student_network.state_dict()[
                        'output_layer.weight'
                        ].data
                student_layer_gradients = \
                    student_network.output_layer.weight.grad
            elif self._learner_configuration == "continual":
                student_layer = \
                    student_network.state_dict()[
                        'heads.{}.weight'.format(str(head))
                        ].data
                student_layer_gradients = \
                    student_network.heads[int(head)].weight.grad
            teacher_layers = [
                teacher.state_dict()['output_layer.weight'].data
                for teacher in teacher_networks
                ]
            layer = layer + "_head_{}".format(str(head))

        return layer, student_layer, student_layer_gradients, teacher_layers

    def _compute_overlap_differences(
        self,
        layer_overlaps: np.ndarray
    ) -> np.ndarray:
        overlap_differences = np.zeros((
            len(layer_overlaps), len(layer_overlaps[0]),
            len(layer_overlaps), len(layer_overlaps[0])
        ))
        for l1 in range(len(layer_overlaps)):
            for l2 in range(len(layer_overlaps[l1])):
                it = layer_overlaps[l1][l2]
                for i in range(len(layer_overlaps)):
                    for j in range(len(layer_overlaps[l2])):
                        it2 = layer_overlaps[i][j]
                        overlap_differences[l1, l2, i, j] = \
                            it - it2
        return overlap_differences

    def _compute_student_grad_student_overlap(
        self,
        student_layer: torch.Tensor,
        student_layer_gradients: torch.Tensor
    ) -> np.ndarray:
        student_grad_student_overlap = (
            student_layer_gradients.mm(student_layer.t()) /
            self._input_dimension
            ).cpu().numpy()
        return student_grad_student_overlap

    def _compute_student_self_overlap(
        self,
        student_layer: torch.Tensor
    ) -> np.ndarray:
        # compute overlap matrices
        student_self_overlap = (
            student_layer.mm(student_layer.t()) / self._input_dimension
            ).cpu().numpy()
        return student_self_overlap

    def _compute_student_teacher_overlaps(
        self,
        student_layer: torch.Tensor,
        teacher_layers: List[torch.Tensor],
        head: Union[None, int]
    ) -> List[np.ndarray]:
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

        return student_teacher_overlaps

    def _compute_teacher_self_overlaps(
        self,
        teacher_layers: List[torch.Tensor]
    ) -> List[np.ndarray]:
        teacher_self_overlaps = [
                (
                    teacher_layer.mm(teacher_layer.t()) / self._input_dimension
                ).cpu().numpy()
                for teacher_layer in teacher_layers
                ]
        return teacher_self_overlaps

    def _compute_teacher_teacher_overlaps(
        self,
        teacher_layers: List[torch.Tensor]
    ) -> Dict[Tuple[int, int], np.ndarray]:
        teacher_pairs = list(
            itertools.combinations(range(len(teacher_layers)), 2)
            )
        teacher_teacher_overlaps = {(i, j): (
            teacher_layers[i].mm(teacher_layers[j].t()) / self._input_dimension
            ).cpu().numpy() for (i, j) in teacher_pairs
            }
        return teacher_teacher_overlaps

    # log overlap values (scalars vs image graphs below)
    def _log_matrix_values(
        self,
        layer: str,
        step_count: int,
        log_name: str,
        matrix: np.ndarray
    ) -> None:
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

    def _log_difference_matrix_values(
        self,
        layer: str,
        step_count: int,
        log_name: str,
        matrix: np.ndarray
    ) -> None:
        matrix_shape = matrix.shape
        for i in range(matrix_shape[0]):
            for j in range(matrix_shape[1]):
                for k in range(matrix_shape[2]):
                    for m in range(matrix_shape[3]):
                        if self._verbose_tb > 1:
                            self._writer.add_scalar(
                                "layer_{}_{}/difference_{}_{}_vs_{}_{}".format(
                                    layer, log_name, i, j, k, m
                                    ),
                                matrix[i][j][k][m],
                                step_count
                                )
                        if self._log_to_df:
                            self._logger_df.at[
                                step_count,
                                "layer_{}_{}/difference_{}_{}_vs_{}_{}".format(
                                    layer, log_name, i, j, k, m
                                    )
                                ] = matrix[i][j][k][m]
