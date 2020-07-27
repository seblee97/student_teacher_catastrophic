import os
import warnings
from typing import List
from typing import Optional

import numpy as np

from ode.configuration import StudentTwoTeacherConfiguration
from ode.integrals import Integrals
from ode.plotter import Plotter


class StudentTeacherODE:

    def __init__(self, overlap_configuration: StudentTwoTeacherConfiguration, nonlinearity: str,
                 w_learning_rate: float, h_learning_rate: float, curriculum: List[int],
                 soft_committee: bool):

        self._configuration = overlap_configuration
        self._nonlinearity = nonlinearity
        self._w_learning_rate = w_learning_rate
        self._h_learning_rate = h_learning_rate
        self._soft_committee = soft_committee

        if curriculum is not None:
            self._curriculum = iter(curriculum)
            try:
                self._next_switch_step = next(self._curriculum)
            except StopIteration:
                self._next_switch_step = np.inf
        else:
            self._curriculum = None

        self._active_teacher = 0

        if nonlinearity == "relu":
            self._i2_fn = Integrals.relu_i2
            self._i3_fn = Integrals.relu_i3
            self._i4_fn = Integrals.relu_i4
        elif (nonlinearity == "sigmoid") or (nonlinearity == "scaled_erf"):
            self._i2_fn = Integrals.sigmoid_i2
            self._i3_fn = Integrals.sigmoid_i3
            self._i4_fn = Integrals.sigmoid_i4

        self._timestep = 0
        self._error_1_log = []
        self._error_2_log = []

    @property
    def configuration(self) -> StudentTwoTeacherConfiguration:
        return self._configuration

    @property
    def timestep(self):
        return self._timestep

    @property
    def error_1_log(self):
        return self._error_1_log

    @property
    def error_2_log(self):
        return self._error_1_log

    @property
    def dr_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            teacher_index = "m"
            student_head = self._configuration.h1
        else:
            teacher_head = self._configuration.th2
            teacher_index = "p"
            student_head = self._configuration.h2
        derivative = np.zeros(self._configuration.R.shape).astype(float)
        for (i, n), _ in np.ndenumerate(derivative):
            in_derivative = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n),
                                                                      (teacher_index, m)])
                in_derivative += head_unit * self._i3_fn(cov)
            for k, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n), ("k", k)])
                in_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][n] = self._w_learning_rate * student_head[i] * in_derivative

        return derivative

    @property
    def dq_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            teacher_index = "m"
            teacher_index_2 = "n"
            student_head = self._configuration.h1
        else:
            teacher_head = self._configuration.th2
            teacher_index = "p"
            teacher_index_2 = "q"
            student_head = self._configuration.h2
        derivative = np.zeros(self._configuration.Q.shape).astype(float)
        for (i, k), _ in np.ndenumerate(derivative):
            ik_derivative = 0

            sum_1 = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k),
                                                                      (teacher_index, m)])
                sum_1 += head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("j", j)])
                sum_1 -= head_unit * self._i3_fn(cov)

            ik_derivative += self._w_learning_rate * student_head[i] * sum_1

            sum_2 = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix([("k", k), ("i", i),
                                                                      (teacher_index, m)])
                sum_2 += head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix([("k", k), ("i", i), ("j", j)])
                sum_2 -= head_unit * self._i3_fn(cov)

            ik_derivative += self._w_learning_rate * student_head[k] * sum_2

            sum_3 = 0
            for j, head_unit_j in enumerate(student_head):
                for l, head_unit_l in enumerate(student_head):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k),
                                                                          ("j", j), ("l", l)])
                    sum_3 += head_unit_j * head_unit_l * self._i4_fn(cov)
            for m, head_unit_m in enumerate(teacher_head):
                for n, head_unit_n in enumerate(teacher_head):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k),
                                                                          (teacher_index, m),
                                                                          (teacher_index_2, n)])
                    sum_3 += head_unit_m * head_unit_n * self._i4_fn(cov)
            for m, head_unit_m in enumerate(teacher_head):
                for j, head_unit_j in enumerate(student_head):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k),
                                                                          ("j", j),
                                                                          (teacher_index, m)])
                    sum_3 -= 2 * head_unit_m * head_unit_j * self._i4_fn(cov)

            ik_derivative += self._w_learning_rate**2 * student_head[i] * student_head[k] * sum_3

            derivative[i][k] = ik_derivative

        return derivative

    @property
    def du_dt(self) -> np.ndarray:
        if self._active_teacher == 0:
            teacher_head = self._configuration.th1
            teacher_index = "m"
            student_head = self._configuration.h1
        else:
            teacher_head = self._configuration.th2
            teacher_index = "q"
            student_head = self._configuration.h2
        derivative = np.zeros(self._configuration.U.shape).astype(float)
        for (i, p), _ in np.ndenumerate(derivative):
            ip_derivative = 0
            for m, head_unit in enumerate(teacher_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p),
                                                                      (teacher_index, m)])
                ip_derivative += head_unit * self._i3_fn(cov)
            for k, head_unit in enumerate(student_head):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p), ("k", k)])
                ip_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][p] = self._w_learning_rate * student_head[i] * ip_derivative

        return derivative

    @property
    def dh1_dt(self):
        derivative = np.zeros(self._configuration.h1.shape).astype(float)
        if not self._soft_committee and self._active_teacher == 0:
            for i in range(len(derivative)):
                i_derivative = 0
                for m, head_unit in enumerate(self._configuration.th1):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("m", m)])
                    i_derivative += head_unit * self._i2_fn(cov)
                for k, head_unit in enumerate(self._configuration.h1):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k)])
                    i_derivative -= head_unit * self._i2_fn(cov)

                derivative[i] = self._h_learning_rate * i_derivative

        return derivative

    @property
    def dh2_dt(self):
        derivative = np.zeros(self._configuration.h2.shape).astype(float)
        if not self._soft_committee and self._active_teacher == 1:
            for i in range(len(derivative)):
                i_derivative = 0
                for p, head_unit in enumerate(self._configuration.th2):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p)])
                    i_derivative += head_unit * self._i2_fn(cov)
                for k, head_unit in enumerate(self._configuration.h2):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k)])
                    i_derivative -= head_unit * self._i2_fn(cov)

                derivative[i] = self._h_learning_rate * i_derivative

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
                cov = self._configuration.generate_covariance_matrix([("i", i), ("j", j)])
                error += head_unit_i * head_unit_j * self._i2_fn(cov)
        for n, teacher_head_unit_n in enumerate(self._configuration.th1):
            for m, teacher_head_unit_m in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("n", n), ("m", m)])
                error += teacher_head_unit_n * teacher_head_unit_m * self._i2_fn(cov)
        for i, head_unit_i in enumerate(self._configuration.h1):
            for n, teacher_head_unit_n in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n)])
                error -= 2 * head_unit_i * teacher_head_unit_n * self._i2_fn(cov)
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
                "Run 'self.error_1_log' or self.error_2_log' to view error logs to this point.")
            import pdb
            pdb.set_trace()
        return error

    @property
    def error_2(self):
        error = 0
        for i, head_unit_i in enumerate(self._configuration.h2):
            for j, head_unit_j in enumerate(self._configuration.h2):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("j", j)])
                error += head_unit_i * head_unit_j * self._i2_fn(cov)
        for p, teacher_head_unit_p in enumerate(self._configuration.th2):
            for q, teacher_head_unit_q in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix([("p", p), ("q", q)])
                error += teacher_head_unit_p * teacher_head_unit_q * self._i2_fn(cov)
        for i, head_unit_i in enumerate(self._configuration.h2):
            for p, teacher_head_unit_p in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p)])
                error -= 2 * head_unit_i * teacher_head_unit_p * self._i2_fn(cov)
        return error

    def _step(self):
        self._timestep += 1

        q_delta = self.dq_dt
        r_delta = self.dr_dt
        u_delta = self.du_dt
        h1_delta = self.dh1_dt
        h2_delta = self.dh2_dt

        self._configuration.step_Q(q_delta)
        self._configuration.step_R(r_delta)
        self._configuration.step_U(u_delta)
        self._configuration.step_h1(h1_delta)
        self._configuration.step_h2(h2_delta)

        error_1 = self.error_1
        error_2 = self.error_2

        self._error_1_log.append(error_1)
        self._error_2_log.append(error_2)

    def _switch_teacher(self):
        self._active_teacher = int(not self._active_teacher)
        try:
            self._next_switch_step = next(self._curriculum)
        except StopIteration:
            self._next_switch_step = np.inf

    def step(self, num_steps: int):
        for i in range(num_steps):
            if self._curriculum is not None:
                if i == self._next_switch_step:
                    self._switch_teacher()
            if i % 1000 == 0:
                print(f"Step {i} of ODE dynamics")
            self._step()

    def make_plot(self, save_path: Optional[str]):
        error_dict = {
            "Error (Linear)": {
                "T1 Error": self._error_1_log,
                "T2 Error": self._error_2_log
            },
            "Error (Log)": {
                "T1 Error": np.log10(self._error_1_log),
                "T2 Error": np.log10(self._error_2_log)
            }
        }

        overlap_dict = {
            "Q": self._configuration.Q_log,
            "R": self._configuration.R_log,
            "U": self._configuration.U_log,
            "h1": self._configuration.h1_log,
            "h2": self._configuration.h2_log
        }

        fig = Plotter({**error_dict, **overlap_dict}).plot()

        if save_path:
            fig.savefig(os.path.join(save_path, "ode_plot_summary.pdf"), dpi=100)
