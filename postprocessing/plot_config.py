from utils import Parameters
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

        attribute_config = {}

        generalisation_error_keys = [
            "teacher_{}_generalisation_error/log".format(t)
            for t in range(num_teachers)
        ]
        attribute_config["Generalisation Error (Log)"] = \
            {
            "keys": generalisation_error_keys,
            "plot_type": "scalar",
            "labels": [
                "Teacher {}".format(t) for t in range(num_teachers)
                ]
            }

        for t in range(num_teachers):

            st_fill_ranges = [
                range(len(student_hidden_layers[0])),
                range(len(teacher_hidden_layers[0]))
                ]
            st_fill_combos = list(itertools.product(*st_fill_ranges))

            # first layer overlaps; student-teacher
            student_teacher_t_overlaps = [
                "layer_0_student_teacher_overlaps/{}/values_{}_{}".format(
                    t, i, j
                )
                for (i, j) in enumerate(st_fill_combos)
            ]
            attribute_config["Student-Teacher {} Overlaps".format(t)] = \
                {
                "keys": student_teacher_t_overlaps,
                "plot_type": "scalar",
                "labels": []
                }
            attribute_config[
                "Student-Teacher {} Overlaps (Image)".format(t)
                ] = {
                    "keys": student_teacher_t_overlaps,
                    "plot_type": "image",
                    "labels": []
                    }

            # student head weights
            student_head_weights = [
                "student_head_{}_weight_{}".format(t, i)
                for i in range(student_hidden_layers)
            ]
            attribute_config[
                "Student Head Weights (Teacher {})".format(t)
                ] = {
                    "keys": student_head_weights,
                    "plot_type": "scalar",
                    "labels": [
                        "Weight {}".format(h)
                        for h in range(student_hidden_layers)
                        ]
                    }

        ss_fill_ranges = [
            range(len(student_hidden_layers[0])),
            range(len(teacher_hidden_layers[0]))
            ]
        ss_fill_combos = list(itertools.product(*ss_fill_ranges))

        student_student_overlaps = [
            "layer_0_student_self_overlap/values_{}_{}".format(
                i, j
            )
            for (i, j) in enumerate(ss_fill_combos)
        ]
        attribute_config["Student-Student Overlaps"] = \
            {
            "keys": student_student_overlaps,
            "plot_type": "scalar",
            "labels": []
            }

        return attribute_config
