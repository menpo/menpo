import abc
from copy import deepcopy

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

        Returns
        -------
        vector : (N,) ndarray
            The flattened vector.
        """
        pass

    @abc.abstractmethod
    def from_vector_inplace(self, vector):
        """
        Update the state of this object from the provided 1D flattened
        array.

        Parameters
        ----------
        vector : (N,) ndarray
            Flattened representation of the object.
        """
        pass

    def from_vector(self, vector):
        """
        Build a new instance of the object from the provided 1D flattened
        array, using ``self`` to fill out the missing state required to
        rebuild a full object from it's standardized flattened state.

        A default implementation is provided where a deepcopy of the object
        is made followed by an from_vector_inplace(). This method can be
        overridden for a performance benefit if desired.

        Parameters
        ----------
        vector : (N,) ndarray
            Flattened representation of the object.

        Returns
        -------
        object : :class:`Vectorizable`
            An instance of the class.
        """
        self_copy = deepcopy(self)
        self_copy.from_vector_inplace(vector)
        return self_copy
