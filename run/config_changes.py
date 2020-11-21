"""Mapping from configuration attributes to values

For a given experiment we may want to compare several
values of a given configuration attribute while keeping
everything else the same. Rather than write many
configuration files we can use the same base for all and
systematically modify it for each different run.
"""

import numpy as np


class ConfigChange:

    config_changes = {
        "v_0": [("feature_rotation_magnitude", np.arccos(0))],
        "v_0.2": [("feature_rotation_magnitude", np.arccos(0.2))],
        "v_0.4": [("feature_rotation_magnitude", np.arccos(0.4))],
        "v_0.6": [("feature_rotation_magnitude", np.arccos(0.6))],
        "v_0.8": [("feature_rotation_magnitude", np.arccos(0.8))],
        "v_1": [("feature_rotation_magnitude", np.arccos(1.0))],
    }
