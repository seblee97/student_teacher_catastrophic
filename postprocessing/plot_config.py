from utils import Parameters
from constants import Constants
from typing import Dict, List

import itertools


class PlotConfigGenerator:

    SHADES = [Constants.TORQUOISE_SHADES, Constants.ORANGE_SHADES]

    @staticmethod
    def generate_plotting_config(config: Parameters) -> Dict[str, Dict]:

        num_teachers = config.get(["task", "num_teachers"])
        student_hidden_layers = \
            config.get(["model", "student_hidden_layers"])
        teacher_hidden_layers = \
            config.get(["model", "teacher_hidden_layers"])

        assert len(Constants.TEACHER_SHADES) >= num_teachers, \
            (
                "Not enough shades given for teachers."
                "{} given, {} needed".format(
                    len(Constants.TEACHER_SHADES), num_teachers
                    )
            )

        attribute_config = {
            **PlotConfigGenerator._get_log_genealisation_error_config(
                num_teachers
                ),
            **PlotConfigGenerator._get_linear_genealisation_error_config(
                num_teachers
                ),
            **PlotConfigGenerator._get_student_teacher_overlap_config(
                num_teachers, student_hidden_layers, teacher_hidden_layers
                ),
            **PlotConfigGenerator._get_student_heads_config(
                num_teachers, student_hidden_layers
                ),
            **PlotConfigGenerator._get_student_self_overlap_config(
                student_hidden_layers
                ),
            **PlotConfigGenerator._get_student_grad_self_overlap_config(
                student_hidden_layers
                )
            }

        return attribute_config

    @staticmethod
    def _get_log_genealisation_error_config(
        num_teachers: int
    ) -> Dict[str, Dict]:

        attribute_config = {}

        generalisation_error_log_keys = [
            "teacher_{}_generalisation_error/log".format(t)
            for t in range(num_teachers)
        ]
        attribute_config["Generalisation Error (Log)"] = \
            {
            "keys": generalisation_error_log_keys,
            "plot_type": "scalar",
            "labels": [
                "Teacher {}".format(t) for t in range(num_teachers)
                ],
            "colours": Constants.TEACHER_SHADES
            }
        return attribute_config

    @staticmethod
    def _get_linear_genealisation_error_config(
        num_teachers: int
    ) -> Dict[str, Dict]:

        attribute_config = {}

        generalisation_error_keys = [
            "teacher_{}_generalisation_error/linear".format(t)
            for t in range(num_teachers)
        ]
        attribute_config["Generalisation Error"] = \
            {
            "keys": generalisation_error_keys,
            "plot_type": "scalar",
            "labels": [
                "Teacher {}".format(t) for t in range(num_teachers)
                ],
            "colours": Constants.TEACHER_SHADES
            }
        return attribute_config

    @staticmethod
    def _get_student_teacher_overlap_config(
        num_teachers: int,
        student_hidden_layers: List[int],
        teacher_hidden_layers: List[int]
    ) -> Dict[str, Dict]:

        attribute_config = {}

        for t in range(num_teachers):

            st_fill_ranges = [
                range(student_hidden_layers[0]),
                range(teacher_hidden_layers[0])
                ]
            st_fill_combos = list(itertools.product(*st_fill_ranges))

            # first layer overlaps; student-teacher
            student_teacher_t_overlaps = [
                "layer_0_student_teacher_overlaps/{}/values_{}_{}".format(
                    t, i, j
                )
                for (i, j) in st_fill_combos
            ]
            attribute_config["Student-Teacher {} Overlaps".format(t)] = \
                {
                "keys": student_teacher_t_overlaps,
                "plot_type": "scalar",
                "labels": [
                    r"$R_{{{0}{1}}}$".format(i, j)
                    for (i, j) in st_fill_combos
                ],
                "colours": PlotConfigGenerator.SHADES[t][:len(st_fill_combos)]
                }
            attribute_config[
                "Student-Teacher {} Overlaps (Image)".format(t)
                ] = {
                    "keys": {
                        (j, k): student_teacher_t_overlaps[i]
                        for i, (j, k) in enumerate(st_fill_combos)
                        },
                    "plot_type": "image",
                    "key_format_ranges": [
                        student_hidden_layers[0],
                        teacher_hidden_layers[0]
                        ],
                    "labels": [],
                    }

        return attribute_config

    @staticmethod
    def _get_student_heads_config(
        num_teachers: int,
        student_hidden_layers: List[int],
    ) -> Dict[str, Dict]:

        attribute_config = {}

        for t in range(num_teachers):
            # student head weights
            student_head_weights = [
                "student_head_{}_weight_{}".format(t, i)
                for i in range(student_hidden_layers[-1])
            ]
            attribute_config[
                "Student Head Weights (Teacher {})".format(t)
                ] = {
                    "keys": student_head_weights,
                    "plot_type": "scalar",
                    "labels": [
                        "Weight {}".format(h)
                        for h in range(student_hidden_layers[-1])
                        ],
                    "colours": PlotConfigGenerator.SHADES[t][
                        :student_hidden_layers[-1]
                        ]
                    }

        return attribute_config

    @staticmethod
    def _get_student_self_overlap_config(
        student_hidden_layers: List[int],
    ) -> Dict[str, Dict]:

        attribute_config = {}

        ss_fill_ranges = [
            range(student_hidden_layers[0]),
            range(student_hidden_layers[0])
            ]
        ss_fill_combos = list(itertools.product(*ss_fill_ranges))

        student_student_overlaps = [
            "layer_0_student_self_overlap/values_{}_{}".format(
                i, j
            )
            for (i, j) in ss_fill_combos
        ]
        attribute_config["Student-Student Overlaps"] = \
            {
            "keys": student_student_overlaps,
            "plot_type": "scalar",
            "labels": [
                r"$Q_{{{0}{1}}}$".format(i, j)
                for (i, j) in ss_fill_combos
            ],
            "colours": Constants.STUDENT_SHADES[:len(ss_fill_combos)]
            }

        return attribute_config

    @staticmethod
    def _get_student_grad_self_overlap_config(
        student_hidden_layers: List[int],
    ) -> Dict[str, Dict]:

        attribute_config = {}

        ss_fill_ranges = [
            range(student_hidden_layers[0]),
            range(student_hidden_layers[0])
            ]
        ss_fill_combos = list(itertools.product(*ss_fill_ranges))

        student_grad_student_overlaps = [
            "layer_0_student_grad_student_overlap/values_{}_{}".format(
                i, j
            )
            for (i, j) in ss_fill_combos
        ]
        attribute_config["Gradient-Student Overlaps"] = \
            {
            "keys": student_grad_student_overlaps,
            "plot_type": "scalar",
            "labels": [
                r"$\nabla$" + r"$Q_{{{0}{1}}}$".format(i, j)
                for (i, j) in ss_fill_combos
            ],
            "colours": Constants.STUDENT_SHADES[:len(ss_fill_combos)],
            "smoothing": 50
            }

        return attribute_config
