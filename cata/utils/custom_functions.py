import itertools
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from scipy import stats


def _generate_rotated_vectors(
    dimension: int, theta: float, normalisation: Union[float, int] = 1
) -> Tuple[np.ndarray]:
    """
    Generate 2 N-dimensional vectors that are rotated by an angle theta from each other.

    Args:
        dimension: desired dimension of two vectors.
        theta: angle to be rotated (in radians).
        normalisation: scaling of final two vectors.
            e.g. with normalisation 1, y_1 cdot y_2 = 1.

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


def generate_rotated_vectors(
    dimension: int, theta: float, normalisation: Union[float, int] = 1
) -> Tuple[np.ndarray]:
    """
    Generate 2 N-dimensional vectors that are rotated by an angle theta from each other.

    Args:
        dimension: desired dimension of two vectors.
        theta: angle to be rotated (in radians).
        normalisation: scaling of final two vectors.
            e.g. with normalisation 1, y_1 cdot y_2 = 1.

    Returns:
        rotated_vectors: tuple of two vectors appropriately rotated.
    """
    v_1 = np.random.normal(size=(dimension))
    v_2 = np.random.normal(size=(dimension))
    normal_1 = normalisation * v_1 / np.linalg.norm(v_1)
    normal_2 = normalisation * v_2 / np.linalg.norm(v_2)

    stacked_orthonormal = np.stack((normal_1, normal_2)).T

    # generate rotation vectors
    x_1 = np.array([0, 1])
    x_2 = np.array([np.sin(theta), np.cos(theta)])

    # generate rotated vectors
    y_1 = np.dot(stacked_orthonormal, x_1)
    y_2 = np.dot(stacked_orthonormal, x_2)

    return y_1, y_2


def _generate_rotated_matrices(
    unrotated_weights: torch.Tensor,
    alpha: float,
    normalisation: Union[None, int],
):
    if normalisation is not None:
        # orthonormalise
        self_overlap = torch.mm(unrotated_weights, unrotated_weights.T) / normalisation
        L = torch.cholesky(self_overlap)
        orthonormal_weights = torch.mm(torch.inverse(L), unrotated_weights)
    else:
        orthonormal_weights = unrotated_weights

    second_random_matrix = torch.randn(orthonormal_weights.shape)

    # if normalisation is not None:
    #     second_random_matrix = (
    #         np.sqrt(normalisation)
    #         * (second_random_matrix.T / torch.norm(second_random_matrix, dim=1)).T
    #     )

    second_teacher_rotated_weights = (
        alpha * orthonormal_weights + np.sqrt(1 - alpha ** 2) * second_random_matrix
    )

    import pdb

    pdb.set_trace()

    return orthonormal_weights, second_teacher_rotated_weights


def generate_rotated_matrices(
    unrotated_weights: torch.Tensor,
    alpha: float,
    normalisation: Union[None, float, int] = None,
    orthogonalise: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Method for generating two sets of matrices 'rotated' by a specified amount.
    Here 'rotation' corresponds to an interpolation between an original matrix, W_1 (provided),
    and an independently generated (orthogonal in the limit) matrix of the same dimension.
    This interpolation (alpha) is equivalent (in the limit) to the sum of diagonal elements
    of the overlap matrix of the two matrices W_1^TW_2, hence it can be interpreted as a rotation.

    Args:
        unrotated_weights: first weight matrix.
        alpha: interpolation parameter (between 0 and 1).
        normalisation: optional argument for the magnitude of the weights matrix.
        orthogonalise: whether to orthogonalise initial weight matrix.

    Returns:
        w_1: first weight matrix (equal to unrotated_weights argument
        in absence of additional normalisation and orthogonalisation).
        w_2: second weight matrix "rotated" by alpha.
    """
    if orthogonalise:
        normalisation = normalisation or 1
        first_matrix_self_overlap = (
            torch.mm(unrotated_weights, unrotated_weights.T) / normalisation
        )
        first_matrix_L = torch.cholesky(first_matrix_self_overlap)
        # orthonormal first matrix
        first_matrix = torch.mm(torch.inverse(first_matrix_L), unrotated_weights)
    else:
        first_matrix = unrotated_weights

    first_matrix_norms = torch.norm(first_matrix, dim=1)

    random_matrix = torch.randn(unrotated_weights.shape)

    second_matrix = alpha * first_matrix + np.sqrt(1 - alpha ** 2) * random_matrix

    for node_index, node in enumerate(second_matrix):
        node_norm = torch.norm(node)
        scaling = first_matrix_norms[node_index] / node_norm
        second_matrix[node_index] = scaling * node

    return first_matrix, second_matrix


def create_iterator(offset: int, increments: List[int]):
    class CustomIterator:
        def __init__(self):
            self._increments = increments
            self._increment_index_cycle = itertools.cycle(range(len(increments)))
            self._current_increment = next(self._increment_index_cycle)

        def __iter__(self):
            self.a = offset
            return self

        def __next__(self):
            x = self.a
            self.a += self._increments[self._current_increment]
            self._current_increment = next(self._increment_index_cycle)
            return x

    return iter(CustomIterator())
