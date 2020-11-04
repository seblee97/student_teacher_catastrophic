from typing import Tuple
from typing import Union

import numpy as np
from scipy import stats


def generate_rotated_vectors(
    dimension: int, theta: float, normalisation: Union[float, int] = 1
) -> Tuple[np.ndarray]:
    """
    Generate 2 N-dimensional vectors that are rotated by an angle theta from each other.

    Args:
        dimension: desired dimension of two vectors.
        theta: angle to be rotated (in radians).
        normalisation: scaling of final two vectors.
            e.g. with normalisation 1, y_1 \cdot y_2 = 1.

    Returns:
        rotated_vectors: tuple of two vectors appropriately rotated.
    """
    # generate random orthogonal matrix of dimension N
    R = normalisation * stats.ortho_group.rvs(dimension)

    # generate rotation vectors
    x_1 = np.array([0, 1])
    x_2 = np.array([np.sin(theta), np.cos(theta)])

    # generate rotated vectors
    y_1 = np.dot(R[:, :2], x_1)
    y_2 = np.dot(R[:, :2], x_2)

    return y_1, y_2
