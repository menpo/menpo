import abc
from pybug.base import Vectorizable, Landmarkable


class Shape(Vectorizable, Landmarkable):
    """
    Abstract representation of shape.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        Landmarkable.__init__(self)
