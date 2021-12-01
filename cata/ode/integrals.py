import numpy as np
from cata.ode.covariance import CovarianceMatrix


class Integrals:
    @staticmethod
    def _lambda_0(covariance: CovarianceMatrix):
        l0 = (
            Integrals._lambda_4(covariance) * covariance[2, 3]
            - covariance[1, 2] * covariance[1, 3] * (1 + covariance[0, 0])
            - covariance[0, 2] * covariance[0, 3] * (1 + covariance[1, 1])
            + covariance[0, 1] * covariance[0, 2] * covariance[1, 3]
            + covariance[0, 1] * covariance[0, 3] * covariance[1, 2]
        )
        return l0

    @staticmethod
    def _lambda_1(covariance: CovarianceMatrix):
        l1 = (
            Integrals._lambda_4(covariance) * (1 + covariance[2, 2])
            - covariance[1, 2] ** 2 * (1 + covariance[0, 0])
            - covariance[0, 2] ** 2 * (1 + covariance[1, 1])
            + 2 * covariance[0, 1] * covariance[0, 2] * covariance[1, 2]
        )
        return l1

    @staticmethod
    def _lambda_2(covariance: CovarianceMatrix):
        l2 = (
            Integrals._lambda_4(covariance) * (1 + covariance[3, 3])
            - covariance[1, 3] ** 2 * (1 + covariance[0, 0])
            - covariance[0, 3] ** 2 * (1 + covariance[1, 1])
            + 2 * covariance[0, 1] * covariance[0, 3] * covariance[1, 3]
        )
        return l2

    @staticmethod
    def _lambda_3(covariance: CovarianceMatrix):
        l3 = (1 + covariance[0, 0]) * (1 + covariance[2, 2]) - covariance[0, 2] ** 2
        return l3

    @staticmethod
    def _lambda_4(covariance: CovarianceMatrix):
        l4 = (1 + covariance[0, 0]) * (1 + covariance[1, 1]) - covariance[0, 1] ** 2
        return l4

    @staticmethod
    def sigmoid_i2(covariance: CovarianceMatrix):
        nom = covariance[0, 1]
        den = np.sqrt(1 + covariance[0, 0]) * np.sqrt(1 + covariance[1, 1])
        if abs(nom / den) > 1:
            import pdb

            pdb.set_trace()
        return 2 * np.arcsin(nom / den) / np.pi

    @staticmethod
    def relu_i2(covariance: CovarianceMatrix):
        nom_1 = 2 * np.sqrt(covariance[0, 0] * covariance[1, 1] - covariance[0, 1] ** 2)
        nom_2 = np.pi * covariance[0, 1]
        nom_3 = (
            2
            * covariance[0, 1]
            * np.arctan(
                covariance[0, 1]
                / np.sqrt(covariance[0, 0] * covariance[1, 1] - covariance[0, 1] ** 2)
            )
        )
        den = 8 * np.pi
        return (nom_1 + nom_2 + nom_3) / den

    @staticmethod
    def sigmoid_i3(covariance: CovarianceMatrix):
        nom = 2 * (
            covariance[1, 2] * (1 + covariance[0, 0])
            - covariance[0, 1] * covariance[0, 2]
        )
        den = np.pi * np.sqrt(Integrals._lambda_3(covariance)) * (1 + covariance[0, 0])
        return nom / den

    @staticmethod
    def relu_i3(covariance: CovarianceMatrix):
        t1_nom = covariance[0, 1] * np.sqrt(
            covariance[0, 0] * covariance[2, 2] - covariance[0, 2] ** 2
        )
        t1_den = 2 * np.pi * covariance[0, 0]
        t2_nom = covariance[1, 2] * np.arcsin(
            covariance[0, 2] / np.sqrt(covariance[0, 0] * covariance[2, 2])
        )
        t2_den = 2 * np.pi
        return t1_nom / t1_den + t2_nom / t2_den + 0.25 * covariance[1, 2]

    @staticmethod
    def sigmoid_i4(covariance: CovarianceMatrix):
        nom = 4 * np.arcsin(
            Integrals._lambda_0(covariance)
            / np.sqrt(Integrals._lambda_1(covariance) * Integrals._lambda_2(covariance))
        )
        den = np.pi ** 2 * np.sqrt(Integrals._lambda_4(covariance))
        return nom / den

    @staticmethod
    def relu_i4(covariance: CovarianceMatrix):
        raise NotImplementedError

    @staticmethod
    def j2(covariance: CovarianceMatrix):
        base_term = (
            1
            + covariance[0, 0]
            + covariance[1, 1]
            + covariance[0, 0] * covariance[1, 1]
            - covariance[0, 1] ** 2
        )
        return (2 * base_term ** -0.5) / np.pi
