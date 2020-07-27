from typing import List
from typing import Tuple

import numpy as np


class CovarianceMatrix:

    def __init__(self, matrix_values: np.ndarray, indices: List[Tuple[str, int]]):

        self._matrix = matrix_values.astype(float)
        self._indices = indices

        self._check_determinant_constraint()

    @property
    def matrix(self):
        return self._matrix

    @property
    def indices(self):
        return self._indices

    @property
    def shape(self):
        return self._matrix.shape

    def _check_determinant_constraint(self):
        determinant = np.linalg.det(self._matrix)

        if determinant < 0:
            if int(round(determinant)) == 0:
                pass
            else:
                import pdb
                pdb.set_trace()
                raise AssertionError("Covariance matrix must be positive semi-definite."
                                     "Determinant must be non-negative."
                                     f"The matrix {self._matrix} has determinant {determinant}"
                                     f"Indices for this matrix are {self._indices}")

    def __getitem__(self, key: Tuple[int]):
        return self._matrix[key[0]][key[1]]
