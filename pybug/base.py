import abc


class Vectorizable(object):
    """
    Abstract interface that guarantees subclasses can be flattened and
    restored from flattened (vectorized) representations. Useful for
    statistically analyzing objects, which almost always requires the data
    to be provided as a single vector.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def as_vector(self):
        """
        Returns a flattened representation of the object as a single vector.

        Useful for statistical analysis of the object.
        """
        pass

    @abc.abstractmethod
    def from_vector(self, flattened):
        """
        Build a new instance of the object from the provided 1D flattened
        array, using self to fill out the missing state required to rebuild a
        full object from it's standardized flattened state.

        :param flattened: Flattened representation of the object
        :type flattened: ndarray
        """
        pass
