import abc


class Flattenable(object):
    """
    Abstract interface that guarantees subclasses can be flattened and
    restored from flattened (vectorized) representations. Useful for
    statistically analyzing objects, which almost always requires the data
    to be provided as a single vector.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def as_flattened(self):
        """
        Returns a flattened representation of the object as a single vector.

        Useful for statistical analysis of the object.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_flattened_with_instance(cls, flattened, instance, **kwargs):
        """
        Build an instance of the object from the provided 1D flattened array,
        along with an existing instance of the object that will be pillaged
        for additional required information
        :param flattened: Flattened representation of the object
        :param instance: an instance of the Class. This will be used to fill
         out the missing state required to rebuild a full object from it's
         standardized flattened state.
        """
        pass
