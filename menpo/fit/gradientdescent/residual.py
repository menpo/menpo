import abc


class Residual(object):
    r"""
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def error(self):
        pass

    @abc.abstractproperty
    def error_derivative(self):
        pass

    @abc.abstractproperty
    def d_dp(self):
        pass

    @abc.abstractproperty
    def hessian(self):
        pass


class SSD(Residual):

    type = 'SSD'

    def error(self):
        raise ValueError("Not implemented")

    def error_derivative(self):
        raise ValueError("Not implemented")

    def d_dp(self):
        raise ValueError("Not implemented")

    def hessian(self):
        raise ValueError("Not implemented")


class Robust(Residual):

    def __init__(self):
        raise ValueError("Not implemented")

    def error(self):
        raise ValueError("Not implemented")

    def error_derivative(self):
        raise ValueError("Not implemented")

    def d_dp(self):
        raise ValueError("Not implemented")

    def hessian(self):
        raise ValueError("Not implemented")

    @abc.abstractmethod
    def _weights(self):
        pass


class Fair(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class L1L2(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class GemanMcClure(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class Cauchy(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class Welsch(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class Huber(Robust):

    def _weights(self):
        raise ValueError("Not implemented")


class Turkey(Robust):

    def _weights(self):
        raise ValueError("Not implemented")
