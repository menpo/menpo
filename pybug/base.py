import abc
from pybug.shape.landmarks import LandmarkManager


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
        array,using self to fill out the missing state required to rebuild a
        full object from it's standardized flattened state.
        :param flattened: Flattened representation of the object
        """
        pass


class Landmarkable(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.landmarks = {}

    def add_landmark_set(self, label, landmark_dict):
        if not isinstance(landmark_dict, dict):
            raise ValueError('Landmark set must be of type dictionary')
        else:
            self.landmarks[label] = LandmarkManager(
                self, label, landmark_dict=landmark_dict)

    def get_landmark_set(self, label):
        return self.landmarks[label]