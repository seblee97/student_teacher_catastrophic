import numpy as np
import itertools
import os

from ode.configuration import StudentTwoTeacherConfiguration
from ode.integrals import Integrals
from ode.plotter import Plotter

from typing import List, Tuple, Dict, Optional


class StudentTeacherODE:

    def __init__(
        self, 
        overlap_configuration: StudentTwoTeacherConfiguration, 
        nonlinearity: str, 
        w_learning_rate: float, 
        h_learning_rate: float,
        curriculum: Dict
    ):

        self._configuration = overlap_configuration
        self._nonlinearity = nonlinearity
        self._w_learning_rate = w_learning_rate
        self._h_learning_rate = h_learning_rate

        self._curriculum = curriculum

        if nonlinearity == "relu":
            self._i2_fn = Integrals.relu_i2
            self._i3_fn = Integrals.relu_i3
            self._i4_fn = Integrals.relu_i4
        elif nonlinearity == "sigmoid":
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
        derivative = np.zeros(self._configuration.R.shape).astype(float)
        for (i, n), _ in np.ndenumerate(derivative):
            in_derivative = 0
            for m, head_unit in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n), ("m", m)])
                in_derivative += head_unit * self._i3_fn(cov)
            for k, head_unit in enumerate(self._configuration.h1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n), ("k", k)])
                in_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][n] = self._w_learning_rate * self._configuration.h1[i] * in_derivative

        return derivative

    @property
    def dq_dt(self) -> np.ndarray:
        derivative = np.zeros(self._configuration.Q.shape).astype(float)
        for (i, k), _ in np.ndenumerate(derivative):
            ik_derivative = 0

            sum_1 = 0
            for m, head_unit in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("m", m)])
                sum_1 += head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(self._configuration.h1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("j", j)])
                sum_1 -= head_unit * self._i3_fn(cov)

            ik_derivative += self._w_learning_rate * self._configuration.h1[i] * sum_1

            sum_2 = 0
            for m, head_unit in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("k", k), ("i", i), ("m", m)])
                sum_2 += head_unit * self._i3_fn(cov)
            for j, head_unit in enumerate(self._configuration.h1):
                cov = self._configuration.generate_covariance_matrix([("k", k), ("i", i), ("j", j)])
                sum_2 -= head_unit * self._i3_fn(cov)
            
            ik_derivative += self._w_learning_rate * self._configuration.h1[k] * sum_2

            sum_3 = 0
            for j, head_unit_j in enumerate(self._configuration.h1):
                for l, head_unit_l in enumerate(self._configuration.h1):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("j", j), ("l", l)])
                    sum_3 += head_unit_j * head_unit_l * self._i4_fn(cov)
            for m, head_unit_m in enumerate(self._configuration.th1):
                for n, head_unit_n in enumerate(self._configuration.th1):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("m", m), ("n", n)])
                    sum_3 += head_unit_m * head_unit_n * self._i4_fn(cov)
            for m, head_unit_m in enumerate(self._configuration.th1):
                for j, head_unit_j in enumerate(self._configuration.h1):
                    cov = self._configuration.generate_covariance_matrix([("i", i), ("k", k), ("j", j), ("m", m)])
                    sum_3 -= 2 * head_unit_m * head_unit_j * self._i4_fn(cov)

            ik_derivative += self._w_learning_rate ** 2 * self._configuration.h1[i] * self._configuration.h1[k] * sum_3

            derivative[i][k] = ik_derivative

        return derivative

    @property
    def du_dt(self) -> np.ndarray:
        derivative = np.zeros(self._configuration.U.shape).astype(float)
        for (i, p), _ in np.ndenumerate(derivative):
            ip_derivative = 0
            for m, head_unit in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p), ("m", m)])
                ip_derivative += head_unit * self._i3_fn(cov)
            for k, head_unit in enumerate(self._configuration.h1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p), ("k", k)])
                ip_derivative -= head_unit * self._i3_fn(cov)

            derivative[i][p] = self._w_learning_rate * self._configuration.h2[i] * ip_derivative

        return derivative

    @property
    def dh1_dt(self):
        derivative = np.zeros(self._configuration.h1.shape).astype(float)
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
        raise NotImplementedError

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
                error += 0.5 * head_unit_i * head_unit_j * self._i2_fn(cov)
        for n, teacher_head_unit_n in enumerate(self._configuration.th1):
            for m, teacher_head_unit_m in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("n", n), ("m", m)])
                error += 0.5 * teacher_head_unit_n * teacher_head_unit_m * self._i2_fn(cov)
        for i, head_unit_i in enumerate(self._configuration.h1):
            for n, teacher_head_unit_n in enumerate(self._configuration.th1):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("n", n)])
                error -= head_unit_i * teacher_head_unit_n * self._i2_fn(cov)
        if error < 0:
            import pdb; pdb.set_trace()
        if np.isnan(error):
            import pdb; pdb.set_trace()
        return error

    @property
    def error_2(self):
        error = 0
        for i, head_unit_i in enumerate(self._configuration.h2):
            for j, head_unit_j in enumerate(self._configuration.h2):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("j", j)])
                error += 0.5 * head_unit_i * head_unit_j * self._i2_fn(cov)
        for p, teacher_head_unit_p in enumerate(self._configuration.th2):
            for q, teacher_head_unit_q in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix([("p", p), ("q", q)])
                error += 0.5 * teacher_head_unit_p * teacher_head_unit_q * self._i2_fn(cov)
        for i, head_unit_i in enumerate(self._configuration.h2):
            for p, teacher_head_unit_p in enumerate(self._configuration.th2):
                cov = self._configuration.generate_covariance_matrix([("i", i), ("p", p)])
                error -= head_unit_i * teacher_head_unit_p * self._i2_fn(cov)
        return error

    def _step(self):
        self._timestep += 1

        q_delta = self.dq_dt
        r_delta = self.dr_dt
        u_delta = self.du_dt
        h1_delta = self.dh1_dt

        self._configuration.step_Q(q_delta)
        self._configuration.step_R(r_delta)
        self._configuration.step_U(u_delta)
        self._configuration.step_h1(h1_delta)
        # self._configuration.hstep_2 = self.dh2_dt

        error_1 = self.error_1
        error_2 = self.error_2

        self._error_1_log.append(error_1)
        self._error_2_log.append(error_2)

    def switch_teacher(self):
        raise NotImplementedError

    def step(self, num_steps: int) -> List[float]:
        for i in range(num_steps):
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

        fig = Plotter({
            **error_dict,
            **overlap_dict
        }).plot()

        if save_path:
            fig.savefig(
                os.path.join(save_path, "ode_plot_summary.pdf"), 
                dpi=100
                )