import abc


class Transform(object):
    """
    An abstract representation of any n-dimensional transform.
    Provides a unified interface to apply the transform (:meth:`apply`)
    """
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def apply(self, x):
        """

        :param x:
        :raise:
        """
        raise NotImplementedError