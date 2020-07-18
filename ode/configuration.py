import itertools
import numpy as np

from ode.overlaps import SelfOverlap, CrossOverlap
from ode.covariance import CovarianceMatrix

from typing import List, Tuple, Dict


class StudentTwoTeacherConfiguration:

    def __init__(
        self, 
        Q: np.ndarray,
        R: np.ndarray, 
        U: np.ndarray, 
        T: np.ndarray, 
        S: np.ndarray,
        V: np.ndarray,
        h1: np.ndarray,
        h2: np.ndarray,
        th1: np.ndarray,
        th2: np.ndarray
    ):
        self._Q = SelfOverlap(Q, final=False)
        self._R = CrossOverlap(R, final=False)
        self._U = CrossOverlap(U, final=False)
        self._T = SelfOverlap(T, final=True)
        self._S = SelfOverlap(S, final=True)
        self._V = SelfOverlap(V, final=True)

        self._h1 = h1.astype(float)
        self._h2 = h2.astype(float)

        self._th1 = th1
        self._th2 = th2

        self._student_indices = ["i", "j", "k", "l"]
        self._teacher1_indices = ["m", "n", "o"]
        self._teacher2_indices = ["p", "q", "r"]

        student_student_index_duos = self.generate_index_combos(self._student_indices, self._student_indices)
        student_teacher1_index_duos = self.generate_index_combos(self._student_indices, self._teacher1_indices)
        student_teacher2_index_duos = self.generate_index_combos(self._student_indices, self._teacher2_indices)
        teacher1_teacher2_index_duos = self.generate_index_combos(self._teacher1_indices, self._teacher2_indices)
        teacher1_teacher1_index_duos = self.generate_index_combos(self._teacher1_indices, self._teacher1_indices)
        teacher2_teacher2_index_duos = self.generate_index_combos(self._teacher2_indices, self._teacher2_indices)

        self._index_duo_overlap_map = {
            **{duo: self.Q for duo in student_student_index_duos},
            **{duo: self.R for duo in student_teacher1_index_duos},
            **{duo: self.U for duo in student_teacher2_index_duos},
            **{duo: self.V for duo in teacher1_teacher2_index_duos},
            **{duo: self.T for duo in teacher1_teacher1_index_duos},
            **{duo: self.S for duo in teacher2_teacher2_index_duos},
        }

        self._Q_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(Q)
        }
        self._T_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(T)
        }
        self._R_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(R)
        }
        self._U_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(U)
        }
        self._V_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(V)
        }
        self._S_log = {
            "".join([str(i), str(j)]): [] for (i, j), _ in np.ndenumerate(S)
        }

        self._h1_log = {
            str(i): [] for i in range(len(h1))
        }
        self._h2_log = {
            str(i): [] for i in range(len(h2))
        }

    @staticmethod
    def generate_index_combos(index_set_1: List[str], index_set_2: List[str]) -> List[str]:
        index_duos = list(set([
            "".join(st) for st in itertools.chain(
                itertools.product(index_set_1, index_set_2),
                itertools.product(index_set_2, index_set_1)
                )
        ]))
        return index_duos
    
    def generate_covariance_matrix(self, indices: List[Tuple[str, int]]) -> CovarianceMatrix:
        covariance = np.zeros((len(indices), len(indices)))
        for i, index_i in enumerate(indices):
            for j, index_j in enumerate(indices):
                index_duo = "".join([index_i[0], index_j[0]])
                covariance[i][j] = self._index_duo_overlap_map[index_duo][index_i, index_j]

        return CovarianceMatrix(covariance, indices=indices)

    def _log_overlap(self, values: np.ndarray, log: Dict[str, List]):
        for (i, j), value in np.ndenumerate(values):
            log["".join([str(i), str(j)])].append(value)

    def _log_head(self, values: np.ndarray, log: Dict[str, List]):
        for i, value in enumerate(values):
            log[str(i)].append(value)

    @property
    def Q(self) -> SelfOverlap:
        return self._Q

    def step_Q(self, delta_Q: np.ndarray) -> None:
        self._Q.step(delta_Q)
        self._log_overlap(values=self._Q.values, log=self._Q_log)

    @property
    def Q_log(self):
        return self._Q_log

    @property
    def R(self):
        return self._R

    def step_R(self, delta_R: np.ndarray) -> None:
        self._R.step(delta_R)
        self._log_overlap(values=self._R.values, log=self._R_log)

    @property
    def R_log(self):
        return self._R_log

    @property
    def U(self):
        return self._U

    def step_U(self, delta_U: np.ndarray) -> None:
        self._U.step(delta_U)
        self._log_overlap(values=self._U.values, log=self._U_log)

    @property
    def U_log(self):
        return self._U_log

    @property
    def T(self):
        return self._T

    def step_T(self, delta_T: np.ndarray) -> None:
        self._T.step(delta_T)
        self._log_overlap(values=self._T.values, log=self._T_log)

    @property
    def T_log(self):
        return self._T_log

    @property
    def S(self):
        return self._S

    def step_S(self, delta_S: np.ndarray) -> None:
        self._S.step(delta_S)
        self._log_overlap(values=self._S.values, log=self._S_log)

    @property
    def S_log(self):
        return self._S_log

    @property
    def V(self):
        return self._V

    def step_V(self, delta_V: np.ndarray) -> None:
        self._V.step(delta_V)
        self._log_overlap(values=self._V.values, log=self._V_log)

    @property
    def V_log(self):
        return self._V_log

    @property
    def h1(self):
        return self._h1

    def step_h1(self, delta_h1: np.ndarray) -> None:
        self._h1 += delta_h1
        self._log_head(values=self._h1, log=self._h1_log)

    @property
    def h1_log(self):
        return self._h1_log

    @property
    def h2(self):
        return self._h2

    def step_h2(self, delta_h2: np.ndarray) -> None:
        self._h2 += delta_h2
        self._log_head(values=self._h2, log=self._h2_log)

    @property
    def h2_log(self):
        return self._h2_log

    @property
    def th1(self):
        return self._th1

    @property
    def th2(self):
        return self._th2


class RandomStudentTwoTeacherConfiguration(StudentTwoTeacherConfiguration):

    def __init__(
        self,
        N: int,
        M: int,
        K: int,
        student_weight_initialisation_std: float,
        teacher_weight_initialisation_std: float,
        initialise_student_heads: bool,
        normalise_teachers: bool,
        symmetric_student_initialisation: bool
        ):

        self._N = N
        self._M = M
        self._K = K
        self._student_weight_initialisation_std = student_weight_initialisation_std
        self._teacher_weight_initialisation_std = teacher_weight_initialisation_std
        self._nomalise_teachers = normalise_teachers
        self._initialise_student_heads = initialise_student_heads
        self._symmetric_student_initialisation = symmetric_student_initialisation

        student_weight_vectors, t1, t2, h1, h2, th1, th2 = \
            self._initialise_weights()

        super().__init__(
            Q=self.weight_overlap_matrix(student_weight_vectors, student_weight_vectors, self._N),
            R=self.weight_overlap_matrix(student_weight_vectors, t1, self._N),
            U=self.weight_overlap_matrix(student_weight_vectors, t2, self._N),
            S=self.weight_overlap_matrix(t1, t1, self._N),
            T=self.weight_overlap_matrix(t2, t2, self._N),
            V=self.weight_overlap_matrix(t1, t2, self._N),
            h1=h1,
            h2=h2,
            th1=th1,
            th2=th2
            )

    @staticmethod
    def weight_overlap(w1, w2, N):
        return sum(w1 * w2) / N

    @classmethod
    def weight_overlap_matrix(
        cls, weight_vectors_1: List, weight_vectors_2: List, N: int
    ):
        overlap_matrix = np.zeros(
            shape=(len(weight_vectors_1), len(weight_vectors_2))
            )
        for i, w1 in enumerate(weight_vectors_1):
            for j, w2 in enumerate(weight_vectors_2):
                overlap_matrix[i][j] = cls.weight_overlap(w1, w2, N)

        return overlap_matrix

    def _initialise_random_weight_vector(self, scale, length):
        weight_vector = np.random.normal(
            scale=scale, size=length
            )
        return weight_vector

    def _initialise_random_teacher_vector(self, length: int):
        weight_vector = self._initialise_random_weight_vector(
            scale=self._teacher_weight_initialisation_std, length=length
            )

        if self._nomalise_teachers:
            weight_vector = np.sqrt(length) * weight_vector / np.linalg.norm(weight_vector)

        return weight_vector

    def _initialise_weights(self):
        
        if self._symmetric_student_initialisation:
            student_weight_vector = self._initialise_random_weight_vector(
                    scale=self._student_weight_initialisation_std, length=self._N
                    )
            student_weight_vectors = [
                student_weight_vector for _ in range(self._M)
            ]
        else:
            student_weight_vectors = [
                self._initialise_random_weight_vector(
                    scale=self._student_weight_initialisation_std, length=self._N
                    )
                for _ in range(self._M)
            ]
        
        teacher_1_weight_vectors = [
            self._initialise_random_teacher_vector(length=self._N)
            for _ in range(self._K)
        ]
        
        teacher_2_weight_vectors = [
            self._initialise_random_teacher_vector(length=self._N)
            for _ in range(self._K)
        ]

        if self._initialise_student_heads:
            student_head_1 = self._initialise_random_weight_vector(
                scale=self._student_weight_initialisation_std, length=self._M
                )
            student_head_2 = self._initialise_random_weight_vector(
                scale=self._student_weight_initialisation_std, length=self._M
                )
        else:
            student_head_1 = np.zeros(self._M)
            student_head_2 = np.zeros(self._M)

        teacher_1_head = self._initialise_random_weight_vector(
            scale=self._teacher_weight_initialisation_std, length=self._K
            )
        teacher_2_head = self._initialise_random_weight_vector(
            scale=self._teacher_weight_initialisation_std, length=self._K
            )

        return student_weight_vectors, teacher_1_weight_vectors, \
            teacher_2_weight_vectors, student_head_1, student_head_2, \
            teacher_1_head, teacher_2_head