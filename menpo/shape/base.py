import abc
from menpo.base import Vectorizable
from menpo.landmark import Landmarkable
from menpo.transform.base import Transformable
from menpo.visualize.base import Viewable


class Shape(Vectorizable, Landmarkable, Transformable, Viewable):
    """
    Abstract representation of shape. Shapes are :map:`Transformable`,
    :map:`Vectorizable`, :map:`Landmarkable` and :map:`Viewable`. This base
    class handles transforming landmarks when the shape is transformed.
    Therefore, implementations of :map:`Shape` have to implement the abstract
    :meth:`_transform_self_inplace` method that handles transforming the Shape
    itself.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Shape, self).__init__()

    def _transform_inplace(self, transform):
        """
        Transform the landmarks and the shape itself.

        Parameters
        ----------
        transform : func
            A function to transform the spatial data with

        Returns
        -------
        self : `self`
            A pointer to `self` (the result of :meth:`_transform_self_inplace`).
        """
        self.landmarks._transform_inplace(transform)
        return self._transform_self_inplace(transform)

    @abc.abstractmethod
    def _transform_self_inplace(self, transform):
        """
        Implement this method to transform the concrete implementation of a
        shape. This is then called by the Shape's :meth:`_transform_inplace` method,
        which will have updated the landmarks beforehand.

        Parameters
        ----------
        transform : func
            A function to transform the spatial data with

        Returns
        -------
        self : `self`
            A pointer to `self`.
        """
        pass
