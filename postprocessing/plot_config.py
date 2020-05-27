from utils import Parameters
from constants import Constants
from typing import Dict

import itertools


class PlotConfigGenerator:

    @staticmethod
    def generate_plotting_config(config: Parameters) -> Dict[str, Dict]:

        num_teachers = config.get(["task", "num_teachers"])
        student_hidden_layers = \
            config.get(["model", "student_hidden_layers"])
        teacher_hidden_layers = \
            config.get(["model", "teacher_hidden_layers"])

        teacher_shades = Constants.TEACHER_SHADES
        assert len(teacher_shades) >= num_teachers, \
            (
                "Not enough shades given for teachers."
                "{} given, {} needed".format(
                    len(teacher_shades), num_teachers
                    )
            )

        student_shades = Constants.STUDENT_SHADES
        blue_shades = Constants.TORQUOISE_SHADES
        green_shades = Constants.ORANGE_SHADES

        shades = [blue_shades, green_shades]

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
            "colours": teacher_shades
            }

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
            "colours": teacher_shades
            }

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
                "colours": shades[t][:len(st_fill_combos)]
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
                    "colours": shades[t][:student_hidden_layers[-1]]
                    }

        ss_fill_ranges = [
            range(student_hidden_layers[0]),
            range(teacher_hidden_layers[0])
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
            "colours": student_shades[:len(ss_fill_combos)]
            }

        return attribute_config
