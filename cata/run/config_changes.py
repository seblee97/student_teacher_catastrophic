"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import itertools

import numpy as np

CONFIG_CHANGES_INTERLEAVE = {
    f"interleave_{i}_{j}_{r}": [
        {"curriculum": {"interleave_period": i, "interleave_duration": j}},
        {"model": {"teachers": {"feature_rotation": {"rotation_magnitude": np.arccos(r)}}}},
    ]
    for (i, j, r) in itertools.product(
        [1, 2, 5, 100, 1000, 10000, 100000],
        [1],
        np.linspace(0, 1, 6),
    )
}

CONFIG_CHANGES_IMPORTANCE = {
    f"importance_{i}_{r}": [
        {"training": {"consolidation": {"type": "ewc", "importance": i}}},
        {"model": {"teachers": {"feature_rotation": {"rotation_magnitude": np.arccos(r)}}}},
    ]
    for (i, r) in itertools.product(
        [0, 10, 100, 1000, 10000, 100000, 1000000],
        np.linspace(0, 1, 6),
    )
}

CONFIG_CHANGES_SIMILARITY = {
    f"similarity_{i}_{j}": [
        {"data": {"multiset_gaussian": {"ms_mean": 0,
                                        "ms_variance": 1,
                                        "mask_proportion": i,
                                        "resample_probability": j}}}
    ]
    for (i, j) in itertools.product(
        [x/5 for x in range(6)], repeat=2
    )
    # Sets the mask proportion (i) and resample probability (j) to every possible value between 0 and 1, depending on
    # the iterable. Example: x/5 for x in range(6) produces x = 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0
    # Repeat = 2 means there are 2 dimensions, so we have values: (0.0, 0.0), (0.0, 0.2), ... (0.2, 0.0), ...  e.t.c.

}

CONFIG_CHANGES = {**CONFIG_CHANGES_INTERLEAVE}

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
