import abc
from pybug.base import Vectorizable
from pybug.landmark import Landmarkable
from pybug.transform.base import Transformable


class Shape(Vectorizable, Landmarkable, Transformable):
    """
    Abstract representation of shape.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        Landmarkable.__init__(self)

    def _transform(self, transform):
        """
        Transform the landmarks and the shape itself
        :param transform:
        :return:
        """
        for manager in self.landmarks.values():
            for label, lmarks in manager:
                lmarks._transform_self(transform)
                manager.update_landmarks(label, lmarks)
        return self._transform_self(transform)

    @abc.abstractmethod
    def _transform_self(self, transform):
        """
        Implement this method to transform the concrete implementation of a
        shape. This is then called by the Shape's _transform method, which
        will have updated the landmarks beforehand
        :param transform: A function to transfrom the spatial data with
        :return: A pointer to self
        """
        pass