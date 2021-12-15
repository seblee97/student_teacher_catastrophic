"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import itertools

import numpy as np

CONFIG_CHANGES = {
    f"interleave_{i}_{j}": [
        {"curriculum": {"interleave_period": i, "interleave_duration": j}}
    ]
    for (i, j) in zip(([1, 5, 10, 50, 100], [1, 2, 5, 10]))
}


# class ConfigChange:

#     config_changes = {f"ewc_{i}": [("importance", i)] for i in [0, 0.01, 0.1, 1, 10]}

# feature_rotation_space = np.linspace(0, 1, 50)
# readout_rotation_space = [np.arccos(i) for i in np.linspace(0, 1, 50)]

# search_space = itertools.product(feature_rotation_space, readout_rotation_space)

# config_changes = {
#     f"feature_{a}_readout_{round(b, 4)}": [
#         ("feature_rotation_alpha", a),
#         ("readout_rotation_magnitude", b),
#     ]
#     for (a, b) in search_space
# }

# config_changes = {
#     "v_0": [("feature_rotation_magnitude", np.arccos(0))],
#     "v_0.2": [("feature_rotation_magnitude", np.arccos(0.2))],
#     "v_0.4": [("feature_rotation_magnitude", np.arccos(0.4))],
#     "v_0.6": [("feature_rotation_magnitude", np.arccos(0.6))],
#     "v_0.8": [("feature_rotation_magnitude", np.arccos(0.8))],
#     "v_1": [("feature_rotation_magnitude", np.arccos(1.0))],
# }
