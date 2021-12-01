import copy
import warnings
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from cata.ode.configuration import StudentTwoTeacherConfiguration
from cata.ode.integrals import Integrals


class StudentTeacherODE:
    def __init__(
        self,
        overlap_configuration: StudentTwoTeacherConfiguration,
        nonlinearity: str,
        w_learning_rate: float,
        h_learning_rate: float,
        dt: Union[float, int],
        soft_committee: bool,
        train_first_layer: bool,
        train_head_layer: bool,
        frozen_feature: bool,
        noise_stds: List[Union[float, int]],
        copy_head_at_switch: bool,
        importance: Optional[Union[int, float]] = None,
    ):

        self._configuration = overlap_configuration
        self._nonlinearity = nonlinearity
        self._w_learning_rate = w_learning_rate
        self._h_learning_rate = h_learning_rate
        self._dt = dt
        self._soft_committee = soft_committee
        self._train_first_layer = train_first_layer
        self._train_head_layer = train_head_layer
        self._frozen_feature = frozen_feature
        self._noise_stds = noise_stds
        self._copy_head_at_switch = copy_head_at_switch
        self._importance = importance

        self._configuration_log = []

        self._frozen = False
        self._num_switches = 0

        # if curriculum is not None:
        #     self._curriculum = iter(curriculum)
        #     try:
        #         self._next_switch_step = next(self._curriculum)
        #     except StopIteration:
        #         self._next_switch_step = np.inf
        # else:
        #     self._curriculum = None

        self._active_teacher = 0

        if nonlinearity == "relu":
            self._i2_fn = Integrals.relu_i2
            self._i3_fn = Integrals.relu_i3
            self._i4_fn = Integrals.relu_i4
        elif (nonlinearity == "sigmoid") or (nonlinearity == "scaled_erf"):
            self._i2_fn = Integrals.sigmoid_i2
            self._i3_fn = Integrals.sigmoid_i3
            self._i4_fn = Integrals.sigmoid_i4

        self._j2_fn = Integrals.j2

        self._setup_log_data_structures()

    def _setup_log_data_structures(self) -> None:

        self._time = 0
        self._step_count = 0
        self._error_1_log = []
        self._error_2_log = []

        self._task_switch_error_1_log = {}
        self._task_switch_error_2_log = {}

        # self._configuration.Q.shape[0]
        self._teacher_1_offset = self._configuration.R.shape[0]
        # self._configuration.Q.shape[0] + self._configuration.R.shape[1]
        self._teacher_2_offset = (
            self._configuration.R.shape[1] + self._configuration.R.shape[0]
        )

        self._consolidation_offset = (
            self._configuration.Q.shape[0]
            + self._configuration.R.shape[1]
            + self._configuration.U.shape[1]
        )

    @property
    def active_teacher(self):
        return self._active_teacher

    @property
    def configuration(self) -> StudentTwoTeacherConfiguration:
        return self._configuration.configuration

    @property
    def step_count(self):
        return self._step_count

    @property
    def error_1_log(self):
        return self._error_1_log

    @property
    def error_2_log(self):
        return self._error_2_log

    @property
    def current_teacher_error(self) -> float:
        if self._active_teacher == 0:
            if self._error_1_log:
                return self._error_1_log[-1]
            else:
                return np.nan
        elif self._active_teacher == 1:
            if self._error_2_log:
                return self._error_2_log[-1]
            else:
                return np.nan

    @property
    def dr_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            student_head = self._configuration.h1
            offset = self._teacher_1_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h2
        else:
            teacher_head = self._configuration.th2
            student_head = self._configuration.h2
            offset = self._teacher_2_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h1
        derivative = np.zeros(self._configuration.R.shape).astype(float)
        for (i, n), _ in np.ndenumerate(derivative):
            in_derivative = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_1_offset + n, offset + m]
                )
                in_derivative += head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_1_offset + n, j]
                )
                in_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][n] = (
                self._dt * self._w_learning_rate * student_head[i] * in_derivative
            )

        if self._importance is not None and self._num_switches == 1:
            for (i, n), _ in np.ndenumerate(derivative):
                derivative[i][n] -= (
                    self._dt
                    * self._w_learning_rate
                    * self._importance
                    * inactive_student_head[i] ** 2
                    * (
                        self._configuration.R.values[i][n]
                        - self._configuration_log[-1].R.values[i][n]
                    )
                )

        return derivative

    @property
    def du_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            student_head = self._configuration.h1
            offset = self._teacher_1_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h2
        else:
            teacher_head = self._configuration.th2
            student_head = self._configuration.h2
            offset = self._teacher_2_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h1
        derivative = np.zeros(self._configuration.U.shape).astype(float)
        for (i, p), _ in np.ndenumerate(derivative):
            ip_derivative = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_2_offset + p, offset + m]
                )
                ip_derivative += head_unit * self._i3_fn(cov)
            for k, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_2_offset, k]
                )
                ip_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][p] = (
                self._dt * self._w_learning_rate * student_head[i] * ip_derivative
            )

        if self._importance is not None and self._num_switches == 1:
            for (i, p), _ in np.ndenumerate(derivative):
                derivative[i][p] -= (
                    self._dt
                    * self._w_learning_rate
                    * self._importance
                    * inactive_student_head[i] ** 2
                    * (
                        self._configuration.U.values[i][p]
                        - self._configuration_log[-1].U.values[i][p]
                    )
                )

        return derivative

    @property
    def dq_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            student_head = self._configuration.h1
            offset = self._teacher_1_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h2
        else:
            teacher_head = self._configuration.th2
            student_head = self._configuration.h2
            offset = self._teacher_2_offset
            if self._importance is not None:
                inactive_student_head = self._configuration.h1
        derivative = np.zeros(self._configuration.Q.shape).astype(float)
        i_range, k_range = derivative.shape
        for i in range(i_range):
            for k in range(i, k_range):
                ik_derivative = 0

                sum_1 = 0
                for m, head_unit in enumerate(teacher_head):
                    cov = self._configuration.generate_covariance_matrix(
                        [i, k, offset + m]
                    )
                    sum_1 += head_unit * student_head[i] * self._i3_fn(cov)
                    cov = self._configuration.generate_covariance_matrix(
                        [k, i, offset + m]
                    )
                    sum_1 += head_unit * student_head[k] * self._i3_fn(cov)
                for j, head_unit in enumerate(student_head):
                    cov = self._configuration.generate_covariance_matrix([i, k, j])
                    sum_1 -= head_unit * student_head[i] * self._i3_fn(cov)
                    cov = self._configuration.generate_covariance_matrix([k, i, j])
                    sum_1 -= head_unit * student_head[k] * self._i3_fn(cov)

                ik_derivative += self._dt * self._w_learning_rate * sum_1

                sum_3 = 0
                for j, head_unit_j in enumerate(student_head):
                    for l, head_unit_l in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, k, j, l]
                        )
                        sum_3 += head_unit_j * head_unit_l * self._i4_fn(cov)
                for m, head_unit_m in enumerate(teacher_head):
                    for n, head_unit_n in enumerate(teacher_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, k, offset + m, offset + n]
                        )
                        sum_3 += head_unit_m * head_unit_n * self._i4_fn(cov)
                for m, head_unit_m in enumerate(teacher_head):
                    for j, head_unit_j in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, k, j, offset + m]
                        )
                        sum_3 -= 2 * head_unit_m * head_unit_j * self._i4_fn(cov)

                # noise term
                cov = self._configuration.generate_covariance_matrix([i, k])
                sum_3 += self._noise_stds[self._active_teacher] ** 2 * self._j2_fn(cov)

                ik_derivative += (
                    self._dt
                    * self._w_learning_rate ** 2
                    * student_head[i]
                    * student_head[k]
                    * sum_3
                )

                derivative[i][k] = ik_derivative

                if self._importance is not None and self._num_switches == 1:
                    # q_ik terms
                    derivative[i][k] += self._configuration.Q.values[i][k] * (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance ** 2
                        * inactive_student_head[i] ** 2
                        * inactive_student_head[k] ** 2
                        - self._dt
                        * self._w_learning_rate
                        * self._importance
                        * inactive_student_head[i] ** 2
                        - self._dt
                        * self._w_learning_rate
                        * self._importance
                        * inactive_student_head[k] ** 2
                    )
                    # q_ik* terms
                    derivative[i][k] += self._configuration.Q_.values[i][k] * (
                        -2
                        * self._dt
                        * self._w_learning_rate ** 2
                        * self._importance ** 2
                        * inactive_student_head[i] ** 2
                        * inactive_student_head[k] ** 2
                        + self._dt
                        * self._w_learning_rate
                        * self._importance
                        * inactive_student_head[i] ** 2
                        + self._dt
                        * self._w_learning_rate
                        * self._importance
                        * inactive_student_head[k] ** 2
                    )
                    # q_ik** terms
                    derivative[i][k] += self._configuration.Q__.values[i][k] * (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance ** 2
                        * inactive_student_head[i] ** 2
                        * inactive_student_head[k] ** 2
                    )
                    # extra integral terms
                    sum_4 = 0
                    for j, head_unit_j in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix([i, j, k])
                        sum_4 += head_unit_j * self._i3_fn(cov)
                    for m, head_unit_m in enumerate(teacher_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, offset + m, k]
                        )
                        sum_4 -= head_unit_m * self._i3_fn(cov)

                    ik_derivative += (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance
                        * student_head[i]
                        * inactive_student_head[k] ** 2
                        * sum_4
                    )

                    sum_5 = 0
                    for j, head_unit_j in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, j, self._consolidation_offset + k]
                        )
                        sum_5 += head_unit_j * self._i3_fn(cov)
                    for m, head_unit_m in enumerate(teacher_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [i, offset + m, self._consolidation_offset + k]
                        )
                        sum_5 -= head_unit_m * self._i3_fn(cov)

                    ik_derivative -= (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance
                        * student_head[i]
                        * inactive_student_head[k] ** 2
                        * sum_5
                    )

                    sum_6 = 0
                    for j, head_unit_j in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix([k, j, i])
                        sum_6 += head_unit_j * self._i3_fn(cov)
                    for m, head_unit_m in enumerate(teacher_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [k, offset + m, i]
                        )
                        sum_6 -= head_unit_m * self._i3_fn(cov)

                    ik_derivative += (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance
                        * student_head[k]
                        * inactive_student_head[i] ** 2
                        * sum_6
                    )

                    sum_7 = 0
                    for j, head_unit_j in enumerate(student_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [k, j, self._consolidation_offset + i]
                        )
                        sum_7 += head_unit_j * self._i3_fn(cov)
                    for m, head_unit_m in enumerate(teacher_head):
                        cov = self._configuration.generate_covariance_matrix(
                            [k, offset + m, self._consolidation_offset + i]
                        )
                        sum_7 -= head_unit_m * self._i3_fn(cov)

                    ik_derivative -= (
                        self._dt
                        * self._w_learning_rate ** 2
                        * self._importance
                        * student_head[k]
                        * inactive_student_head[i] ** 2
                        * sum_7
                    )

        i_lower = np.tril_indices(len(derivative), -1)
        derivative[i_lower] = derivative.T[i_lower]

        return derivative

    @property
    def dh1_dt(self):
        derivative = np.zeros(self._configuration.h1.shape).astype(float)
        if self._train_head_layer and self._active_teacher == 0:
            for i in range(len(derivative)):
                i_derivative = 0
                for m, head_unit in enumerate(self._configuration.th1):
                    cov = self._configuration.generate_covariance_matrix(
                        [i, self._teacher_1_offset + m]
                    )
                    i_derivative += head_unit * self._i2_fn(cov)
                for k, head_unit in enumerate(self._configuration.h1):
                    cov = self._configuration.generate_covariance_matrix([i, k])
                    i_derivative -= head_unit * self._i2_fn(cov)

                derivative[i] = self._dt * self._h_learning_rate * i_derivative

        return derivative

    @property
    def dh2_dt(self):
        derivative = np.zeros(self._configuration.h2.shape).astype(float)
        if self._train_head_layer and self._active_teacher == 1:
            for i in range(len(derivative)):
                i_derivative = 0
                for p, head_unit in enumerate(self._configuration.th2):
                    cov = self._configuration.generate_covariance_matrix(
                        [i, self._teacher_2_offset + p]
                    )
                    i_derivative += head_unit * self._i2_fn(cov)
                for k, head_unit in enumerate(self._configuration.h2):
                    cov = self._configuration.generate_covariance_matrix([i, k])
                    i_derivative -= head_unit * self._i2_fn(cov)

                derivative[i] = self._dt * self._h_learning_rate * i_derivative

        return derivative

    @property
    def dq_star_dt(self):
        derivative = np.zeros(self._configuration.Q_.shape).astype(float)
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            student_head = self._configuration.h1
            if self._importance is not None:
                inactive_student_head = self._configuration.h2
        else:
            teacher_head = self._configuration.th2
            student_head = self._configuration.h2
            if self._importance is not None:
                inactive_student_head = self._configuration.h1
        for (i, k), _ in np.ndenumerate(derivative):
            ik_derivative = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_2_offset + m, self._consolidation_offset + k]
                )
                ik_derivative -= head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix(
                    [i, j, self._consolidation_offset + k]
                )
                ik_derivative += head_unit * self._i3_fn(cov)

            derivative[i][k] = (
                -self._dt * self._w_learning_rate * student_head[i] * ik_derivative
            )

            derivative[i][k] -= (
                self._dt
                * self._w_learning_rate
                * self._importance
                * inactive_student_head[i] ** 2
                * self._configuration.Q_.values[i][k]
            )

            derivative[i][k] += (
                self._dt
                * self._w_learning_rate
                * self._importance
                * inactive_student_head[i] ** 2
                * self._configuration.Q__.values[i][k]
            )

        return derivative

    @property
    def de_active_dt(self):
        raise NotImplementedError

    @property
    def de_passive_dt(self):
        raise NotImplementedError

    @property
    def error_1(self):
        error = 0
        for i, head_unit_i in enumerate(self._configuration.h1):
            for j, head_unit_j in enumerate(self._configuration.h1):
                cov = self._configuration.generate_covariance_matrix([i, j])
                error += 0.5 * head_unit_i * head_unit_j * self._i2_fn(cov)
        for n, teacher_head_unit_n in enumerate(self._configuration.th1):
            for m, teacher_head_unit_m in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix(
                    [self._teacher_1_offset + n, self._teacher_1_offset + m]
                )
                error += (
                    0.5 * teacher_head_unit_n * teacher_head_unit_m * self._i2_fn(cov)
                )
        for i, head_unit_i in enumerate(self._configuration.h1):
            for n, teacher_head_unit_n in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_1_offset + n]
                )
                error -= head_unit_i * teacher_head_unit_n * self._i2_fn(cov)
        if error < 0:
            warnings.warn(
                "Latest error calculation is negative. This could be due to the learning rate being too high, "
                "especially close to convergence. Run 'self.error_1_log' or self.error_2_log' to view error logs to this point."
                "Run self.make_plot(SAVE_PATH) to create plot of errors and overlaps to this point."
            )
            import pdb

            pdb.set_trace()
        if np.isnan(error):
            warnings.warn(
                "Latest error calculation is NaN. "
                "Run 'self.error_1_log' or self.error_2_log' to view error logs to this point."
            )
            import pdb

            pdb.set_trace()
        return error

    @property
    def error_2(self):
        error = 0
        for i, head_unit_i in enumerate(self._configuration.h2):
            for j, head_unit_j in enumerate(self._configuration.h2):
                cov = self._configuration.generate_covariance_matrix([i, j])
                error += 0.5 * head_unit_i * head_unit_j * self._i2_fn(cov)
        for p, teacher_head_unit_p in enumerate(self._configuration.th2):
            for q, teacher_head_unit_q in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix(
                    [self._teacher_2_offset + p, self._teacher_2_offset + q]
                )
                error += (
                    0.5 * teacher_head_unit_p * teacher_head_unit_q * self._i2_fn(cov)
                )
        for i, head_unit_i in enumerate(self._configuration.h2):
            for p, teacher_head_unit_p in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix(
                    [i, self._teacher_2_offset + p]
                )
                error -= head_unit_i * teacher_head_unit_p * self._i2_fn(cov)
        if error < 0:
            warnings.warn(
                "Latest error calculation is negative. This could be due to the learning rate being too high, "
                "especially close to convergence. Run 'self.error_1_log' or self.error_2_log' to view error logs to this point."
                "Run self.make_plot(SAVE_PATH) to create plot of errors and overlaps to this point."
            )
            import pdb

            pdb.set_trace()
        if np.isnan(error):
            warnings.warn(
                "Latest error calculation is NaN. "
                "Run 'self.error_1_log' or self.error_2_log' to view error logs to this point."
            )
            import pdb

            pdb.set_trace()
        return error

    @property
    def next_switch_step(self):
        return self._next_switch_step

    @property
    def time(self):
        return self._time

    def step(self, n_steps: int = 1):

        self._time += self._dt
        self._step_count += n_steps

        self._configuration.step_C()

        error_1 = self.error_1
        error_2 = self.error_2

        self._error_1_log.append(error_1)
        self._error_2_log.append(error_2)

        if self._train_first_layer and not self._frozen:
            q_delta = self.dq_dt
            r_delta = self.dr_dt
            u_delta = self.du_dt
        else:
            q_delta = np.zeros(self._configuration.Q.shape).astype(float)
            r_delta = np.zeros(self._configuration.R.shape).astype(float)
            u_delta = np.zeros(self._configuration.U.shape).astype(float)
        if self._train_head_layer:
            h1_delta = self.dh1_dt
            h2_delta = self.dh2_dt
        else:
            h1_delta = np.zeros(self._configuration.h1.shape).astype(float)
            h2_delta = np.zeros(self._configuration.h2.shape).astype(float)

        if self._importance is not None and self._num_switches == 1:
            q_star_delta = self.dq_star_dt

        self._configuration.step_Q(q_delta)
        self._configuration.step_R(r_delta)
        self._configuration.step_U(u_delta)
        self._configuration.step_h1(h1_delta)
        self._configuration.step_h2(h2_delta)

        if self._importance is not None and self._num_switches == 1:
            self._configuration.step_Q_(q_star_delta)

    def switch_teacher(self):
        # log configuration (useful e.g. for consolidation)
        configuration_copy = copy.deepcopy(self._configuration)
        self._configuration_log.append(configuration_copy)

        new_teacher = int(not self._active_teacher)
        if self._copy_head_at_switch:
            if new_teacher == 0:
                self._configuration.h1 = copy.deepcopy(self._configuration.h2)
            elif new_teacher == 1:
                self._configuration.h2 = copy.deepcopy(self._configuration.h1)
        self._active_teacher = new_teacher
        # try:
        #     self._next_switch_step = next(self._curriculum)
        # except StopIteration:
        #     self._next_switch_step = np.inf
        self._task_switch_error_1_log[self._step_count] = self.error_1
        self._task_switch_error_2_log[self._step_count] = self.error_2
        self._num_switches += 1
        if self._frozen_feature and self._num_switches == 1:
            self._frozen = True

        if self._importance is not None and self._num_switches == 1:
            self._configuration.Q_ = copy.deepcopy(self._configuration.Q).values
            self._configuration.R_ = copy.deepcopy(self._configuration.R).values.T
            self._configuration.U_ = copy.deepcopy(self._configuration.U).values.T
            self._configuration.Q__ = copy.deepcopy(self._configuration.Q).values

    @staticmethod
    def _get_data_diff(data: Union[List, np.ndarray]) -> np.ndarray:
        return np.insert(np.array(data[1:]) - np.array(data[:-1]), 0, 0)
