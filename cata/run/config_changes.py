"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

# import itertools

import numpy as np

CONFIG_CHANGES_OUTPUT_NOISE = {
    f"output_noise_{n}": [
        {
            "model": {
                "teachers": {"teacher_noises": [n, 0]},
                "student": {"student_hidden_layers": [3]},
            }
        }
    ]
    for n in [0, 0.1, 1, 3]
}

CONFIG_CHANGES_INPUT_NOISE = {
    f"input_noise_{n}": [
        {"data": {"noise_to_student": [[0, n], [0, 0]]}},
        {
            "model": {
                "student": {"student_hidden_layers": [3]},
            }
        },
    ]
    for n in [0, 0.01, 0.1, 1, 3]
}

CONFIG_CHANGES_IMPAIRED_UNITS = {
    f"impaired_units_{n}": [
        {
            "model": {
                "student": {"student_hidden_layers": [3]},
            }
        },
        {"training": {"freeze_units": [n, 0]}},
    ]
    for n in [0, 1, 2]
}

CONFIG_CHANGES_OUTPUT_NOISE_OVERPARAM = {
    f"output_noise_{n}": [{"model": {"teachers": {"teacher_noises": [n, 0]}}}]
    for n in [0, 0.1, 1, 3]
}

CONFIG_CHANGES_INPUT_NOISE_OVERPARAM = {
    f"input_noise_{n}": [{"data": {"noise_to_student": [[0, n], [0, 0]]}}]
    for n in [0, 0.01, 0.1, 1, 3]
}

CONFIG_CHANGES_IMPAIRED_UNITS_OVERPARAM = {
    f"impaired_units_{n}": [{"training": {"freeze_units": [n, 0]}}]
    for n in [0, 1, 2, 3, 4, 5]
}

CONFIG_CHANGES = {
    **CONFIG_CHANGES_OUTPUT_NOISE,
    **CONFIG_CHANGES_INPUT_NOISE,
    **CONFIG_CHANGES_OUTPUT_NOISE,
    **CONFIG_CHANGES_OUTPUT_NOISE_OVERPARAM,
    **CONFIG_CHANGES_IMPAIRED_UNITS,
    **CONFIG_CHANGES_IMPAIRED_UNITS_OVERPARAM,
}

# CONFIG_CHANGES = {f"interleave_{i}_feature_{r}": [{"curriculum": {"fixed_period": i}}, {"model": {"teachers": {"feature_rotation": {"rotation_magnitude": np.arccos(r)}}}}] for i, r in itertools.product([1, 2, 10, 100, 1000], np.linspace(0, 1, 11))}


# CONFIG_CHANGES_INTERLEAVE = {
#     f"interleave_{i}_{j}_{r}": [
#         {"curriculum": {"interleave_period": i, "interleave_duration": j}},
#         {"model": {"teachers": {"feature_rotation": {"rotation_magnitude": np.arccos(r)}}}},
#     ]
#     for (i, j, r) in itertools.product(
#         [1, 2, 5, 100, 1000, 10000, 100000],
#         [1],
#         np.linspace(0, 1, 6),
#     )
# }

# CONFIG_CHANGES_IMPORTANCE = {
#     f"importance_{i}_{r}": [
#         {"training": {"consolidation": {"type": "ewc", "importance": i}}},
#         {"model": {"teachers": {"feature_rotation": {"rotation_magnitude": np.arccos(r)}}}},
#     ]
#     for (i, r) in itertools.product(
#         [0, 10, 100, 1000, 10000, 100000, 1000000],
#         np.linspace(0, 1, 6),
#     )
# }

# CONFIG_CHANGES = {**CONFIG_CHANGES_IMPORTANCE, **CONFIG_CHANGES_INTERLEAVE}

# # class ConfigChange:

# #     config_changes = {f"ewc_{i}": [("importance", i)] for i in [0, 0.01, 0.1, 1, 10]}

# # feature_rotation_space = np.linspace(0, 1, 50)
# # readout_rotation_space = [np.arccos(i) for i in np.linspace(0, 1, 50)]

# # search_space = itertools.product(feature_rotation_space, readout_rotation_space)

# # config_changes = {
# #     f"feature_{a}_readout_{round(b, 4)}": [
# #         ("feature_rotation_alpha", a),
# #         ("readout_rotation_magnitude", b),
# #     ]
# #     for (a, b) in search_space
# # }

# # config_changes = {
# #     "v_0": [("feature_rotation_magnitude", np.arccos(0))],
# #     "v_0.2": [("feature_rotation_magnitude", np.arccos(0.2))],
# #     "v_0.4": [("feature_rotation_magnitude", np.arccos(0.4))],
# #     "v_0.6": [("feature_rotation_magnitude", np.arccos(0.6))],
# #     "v_0.8": [("feature_rotation_magnitude", np.arccos(0.8))],
# #     "v_1": [("feature_rotation_magnitude", np.arccos(1.0))],
# # }
