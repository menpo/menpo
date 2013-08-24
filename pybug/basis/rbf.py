import abc
import numpy as np


class BasisFunction(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def phi(self, r):
        pass

    @abc.abstractmethod
    def derivative(self, r):
        pass


class R2LogR2(BasisFunction):

    def __init__(self):
        super(R2LogR2, self).__init__()

    def phi(self, r):
        """
        r^2 log(r^2) === r^2 2log(r)
        :param r:
        :return:
        """
        mask = r == 0
        r[mask] = 1
        U = r ** 2 * (2 * np.log(r))
        # reset singularities to 0
        U[mask] = 0
        return U

    def derivative(self, r):
        """
        2r(log(r^2) + 1) === 2r(2log(r) + 1)
        :param r:
        :return:
        """
        mask = r == 0
        r[mask] = 1
        dUdr = 2 * r * (2 * np.log(r) + 1)
        # reset singularities to 0
        dUdr[mask] = 0
        return dUdr


class R2LogR(BasisFunction):

    def __init__(self):
        super(R2LogR, self).__init__()

    def phi(self, r):
        """
        r^2 log(r)
        :param r:
        :return:
        """
        mask = r == 0
        r[mask] = 1
        U = r ** 2 * np.log(r)
        # reset singularities to 0
        U[mask] = 0
        return U

    def derivative(self, r):
        """
        r(1 + 2log(r))
        :param r:
        :return:
        """
        mask = r == 0
        r[mask] = 1
        dUdr = r * (1 + 2 * np.log(r))
        # reset singularities to 0
        dUdr[mask] = 0
        return dUdr