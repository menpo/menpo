import abc
from copy import deepcopy


class Vectorizable(object):
    """
    Interface that provides methods for 'flattening' an object into a
    vector, and restoring from the same vectorized form. Useful for
    statistical analysis of objects, which commonly requires the data
    to be provided as a single vector.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def n_parameters(self):
        r"""
        The length of the vector that this Vectorizable object produces.

        type: int
        """
        return (self.as_vector()).shape[0]

    @abc.abstractmethod
    def as_vector(self):
        """
        Returns a flattened representation of the object as a single
        vector.

        Returns
        -------
        vector : (N,) ndarray
            The core representation of the object, flattened into a
            single vector.
        """
        pass

    @abc.abstractmethod
    def from_vector_inplace(self, vector):
        """
        Update the state of this object from it's vectorized state

        Parameters
        ----------
        vector : (N,) ndarray
            Flattened representation of this object.
        """
        pass

    def from_vector(self, vector):
        """
        Build a new instance of the object from it's vectorized state.


        ``self`` is used to fill out the missing state required to
        rebuild a full object from it's standardized flattened state. This
        is the default implementation, which is which is a
        ``deepcopy`` of the object followed by a call to
        :meth:`from_vector_inplace()`. This method can be overridden for a
        performance benefit if desired.

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
