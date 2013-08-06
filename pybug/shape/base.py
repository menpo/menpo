import abc
from pybug.shape.landmarks import LandmarkManager
from pybug.base import Vectorizable


class Shape(Vectorizable):
    """ Abstract representation of shape. All subclasses will have some
     data where semantic meaning can be assigned to the i'th item. Shape
     couples a LandmarkManager to this item of data, meaning,
     all subclasses are landmarkable. Note this does not mean that all
     subclasses need have a spatial meaning (i.e. the 7'th node of a Graph
     can still be landmarked)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.landmarks = LandmarkManager(self)
