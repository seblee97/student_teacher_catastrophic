import numpy as np
import itertools

from typing import List, Tuple, Dict


class ODE:

    def __init__(self):
        pass

    @staticmethod
    def lambda_3(covariance: List):
        l3 = (1 + covariance[0][0]) \
            * (1 + covariance[2][2]) \
            - covariance[0][2] ** 2
        return l3

    @staticmethod
    def sigmoid_i3(covariance: List):
        nom = 2 * (
            covariance[1][2] *
            (1 + covariance[0][0]) - covariance[0][1] * covariance[0][2]
            )
        den = np.pi * np.sqrt(ODE.lambda_3(covariance)) \
            * (1 + covariance[0][0])
        return nom / den

    @staticmethod
    def relu_i3(covariance: List):
        t1_nom = covariance[0][1] * np.sqrt(
            covariance[0][0] * covariance[2][2] - covariance[0][2] ** 2
            )
        t1_den = 2 * np.pi * covariance[0][0]
        t2_nom = covariance[1][2] * \
            np.arcsin(covariance[0][2] / np.sqrt(
                covariance[0][0] * covariance[2][2])
                )
        t2_den = 2 * np.pi
        return t1_nom / t1_den + t2_nom / t2_den + 0.25 * covariance[1][2]

    @staticmethod
    def sigmoid_i2():
        raise NotImplementedError

    @staticmethod
    def relu_i2():
        raise NotImplementedError


class StudentTeacherODE:

    def __init__(self, overlaps: Dict[str, np.ndarray], nonlinearity: str, learning_rate: float, teacher_head1: np.ndarray, teacher_head2: np.ndarray):

        self._overlaps = overlaps
        self._nonlinearity = nonlinearity
        self._learning_rate = learning_rate

        self._teacher_head1 = teacher_head1
        self._teacher_head2 = teacher_head2

        if nonlinearity == "relu":
            self._i3_fn = ODE.relu_i3
        elif nonlinearity == "sigmoid":
            self._i3_fn = ODE.sigmoid_i3

        self._student_indices = ["i", "j", "k", "l"]
        self._teacher1_indices = ["m", "n", "o"]
        self._teacher2_indices = ["p", "q", "r"]

        self._id_overlaps = {}

        self._generate_index_combos()

    def _generate_index_combos(self):
        # student-student
        for index_duo in [
            "".join(st) for st in itertools.product(
                self._student_indices, self._student_indices
                )
        ]:
            self._id_overlaps[index_duo] = "Q"

        # student-teacher1
        for index_duo in [
            "".join(st) for st in itertools.chain(
                itertools.product(self._student_indices, self._teacher1_indices),
                itertools.product(self._teacher1_indices, self._student_indices)
            )
        ]:
            self._id_overlaps[index_duo] = "R"

        # student-teacher2
        for index_duo in [
            "".join(st) for st in itertools.chain(
                itertools.product(self._student_indices, self._teacher2_indices),
                itertools.product(self._teacher2_indices, self._student_indices)
            )
        ]:
            self._id_overlaps[index_duo] = "U"

        # teacher1-teacher1
        for index_duo in [
            "".join(st) for st in itertools.product(
                self._teacher1_indices, self._teacher1_indices
                )
        ]:
            self._id_overlaps[index_duo] = "T"

        # teacher2-teacher2
        for index_duo in [
            "".join(st) for st in itertools.product(
                self._teacher2_indices, self._teacher2_indices
                )
        ]:
            self._id_overlaps[index_duo] = "S"

        # teacher1-teacher2
        for index_duo in [
            "".join(st) for st in itertools.chain(
                itertools.product(self._teacher1_indices, self._teacher2_indices),
                itertools.product(self._teacher2_indices, self._teacher1_indices)
            )
        ]:
            self._id_overlaps[index_duo] = "V"

    def _generate_2d_covariance_matrix(self, id1: Tuple[str, int], id2: Tuple[str, int]) -> np.ndarray:
        covariance = np.zeros((2, 2))
        covariance[0][0] = self._overlaps[self._id_overlaps["".join([id1[0], id1[0]])]][id1[1]][id1[1]]
        covariance[0][1] = self._overlaps[self._id_overlaps["".join([id1[0], id2[0]])]][id1[1]][id2[1]]
        covariance[1][0] = self._overlaps[self._id_overlaps["".join([id2[0], id1[0]])]][id2[1]][id1[1]]
        covariance[1][1] = self._overlaps[self._id_overlaps["".join([id2[0], id2[0]])]][id2[1]][id2[1]]
        return covariance

    def _generate_3d_covariance_matrix(self, id1: Tuple[str, int], id2: Tuple[str, int], id3: Tuple[str, int]) -> np.ndarray:
        covariance = np.zeros((3, 3))
        covariance[0][0] = self._overlaps[self._id_overlaps["".join([id1[0], id1[0]])]][id1[1]][id1[1]]
        covariance[0][1] = self._overlaps[self._id_overlaps["".join([id1[0], id2[0]])]][id1[1]][id2[1]]
        covariance[0][2] = self._overlaps[self._id_overlaps["".join([id1[0], id3[0]])]][id1[1]][id3[1]]
        covariance[1][0] = self._overlaps[self._id_overlaps["".join([id2[0], id1[0]])]][id2[1]][id1[1]]
        covariance[1][1] = self._overlaps[self._id_overlaps["".join([id2[0], id2[0]])]][id2[1]][id2[1]]
        covariance[1][2] = self._overlaps[self._id_overlaps["".join([id2[0], id3[0]])]][id2[1]][id3[1]]
        covariance[2][0] = self._overlaps[self._id_overlaps["".join([id3[0], id1[0]])]][id3[1]][id1[1]]
        covariance[2][1] = self._overlaps[self._id_overlaps["".join([id3[0], id2[0]])]][id3[1]][id2[1]]
        covariance[2][2] = self._overlaps[self._id_overlaps["".join([id3[0], id3[0]])]][id3[1]][id3[1]]
        return covariance

    def dr_dt(self) -> np.ndarray:
        derivative = np.zeros_like(self._overlaps["R"]).astype(float)
        for (i, j), _ in np.ndenumerate(derivative):
            s1 = 0
            for m, head_unit in enumerate(self._teacher_head1):
                cov = self._generate_3d_covariance_matrix(("i", i), ("n", j), ("m", m))
                s1 += head_unit * self._i3_fn(cov)

            s2 = 0
            for k, head_unit in enumerate(self._overlaps["h1"]):
                cov = self._generate_3d_covariance_matrix(("i", i), ("n", j), ("k", k))
                s2 += head_unit * self._i3_fn(cov)

            ij_derivative = self._learning_rate * self._overlaps["h1"][i] * (s1 - s2)

            derivative[i][j] = ij_derivative

        return derivative

    def dq_dt(self):
        raise NotImplementedError

    def du_dt(self) -> np.ndarray:
        derivative = np.zeros_like(self._overlaps["U"]).astype(float)
        for (i, p), _ in np.ndenumerate(derivative):
            s1 = 0
            for m, head_unit in enumerate(self._teacher_head1):
                cov = self._generate_3d_covariance_matrix(("i", i), ("p", p), ("m", m))
                s1 += head_unit * self._i3_fn(cov)

            s2 = 0
            for k, head_unit in enumerate(self._overlaps["h1"]):
                cov = self._generate_3d_covariance_matrix(("i", i), ("p", p), ("k", k))
                s2 += head_unit * self._i3_fn(cov)

            ij_derivative = self._learning_rate * self._overlaps["h2"][i] * (s1 - s2)

            derivative[i][p] = ij_derivative

        return derivative

    def dh_dt(self):
        raise NotImplementedError

    def de_active_dt(self):
        raise NotImplementedError

    def de_passive_dt(self):
        raise NotImplementedError
