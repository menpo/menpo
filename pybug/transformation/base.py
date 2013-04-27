class Transformation(object):
    """
    An abstract representation of any n-dimensional transformation.
    Provides a unified interface to apply the transformation (:meth:`apply`)
    """

    def apply(self, x):
        """

        :param x:
        :raise:
        """
        raise NotImplementedError