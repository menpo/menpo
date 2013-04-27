import abc


class Transform(object):
    """
    An abstract representation of any n-dimensional transform.
    Provides a unified interface to apply the transform (:meth:`apply`)
    """
    __metaclass__ = abc.ABCMeta

    def apply(self, x):
        try:
            x._transform(self)
        except AttributeError:
            return self._apply(x)

    @abc.abstractmethod
    def _apply(self, x):
        """
        Applies the transform to the array x, returning the result.
        :param x:
        :raise:
        """
        raise NotImplementedError


class Transformable(object):
    """
    Interface for transformable objects. When Transform.apply() is called on
     an object, if the object has the method _transform,
     the method is called, passing in self (the transform object). This
     allows for the object to define how it should transform itself.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _transform(self, transform):
        """
        Apply the transform given to the Transformable object.
        """
        pass
    