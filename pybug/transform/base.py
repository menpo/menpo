import abc
from pybug.base import Vectorizable
from pybug.visualize.base import Viewable


class Transform(Vectorizable):
    r"""
    An abstract representation of any N-dimensional transform.
    Provides a unified interface to apply the transform (:meth:`apply`). All
    transforms are vectorizable.
    """

    __metaclass__ = abc.ABCMeta

    def apply(self, x, **kwargs):
        r"""
        Applies this transform to ``x``. If ``x`` is :class:`Transformable`,
        ``x`` will be handed this transform object to transform itself
        destructively. If not, ``x`` is assumed to be a numpy array. The
        transformation will be non destructive, returning the transformed
        version. Any ``kwargs`` will be passed to the specific transform
        :meth:`_apply` methods.

        Parameters
        ----------
        x : (N, D) ndarray or an object that implements :class:`Transformable`
            The array or object to be transformed.
        kwargs : dict
            Passed through to :meth:`_apply`.

        Returns
        -------
        transformed : same as ``x``
            The transformed array or object
        """
        def transform(x_):
            """
            Local closure which calls the ``_apply`` method with the ``kwargs``
            attached.
            """
            return self._apply(x_, **kwargs)
        try:
            return x._transform(transform)
        except AttributeError:
            return self._apply(x, **kwargs)

    @abc.abstractmethod
    def _apply(self, x, **kwargs):
        r"""
        Applies the transform to the array ``x``, returning the result.

        Parameters
        ----------
        x : (N, D) ndarray

        Returns
        -------
        transformed : (N, D) ndarray
            Transformed array.
        """
        pass

    @abc.abstractmethod
    def jacobian(self, points):
        r"""
        Calculates the Jacobian of the warp, may be constant.

        Parameters
        ----------
        points : (N, D) ndarray
            The points to calculate the Jacobian over.

        Returns
        -------
        dW_dp : (N, P, D) ndarray
            A (``n_points``, ``n_params``, ``n_dims``) array representing
            the Jacobian of the transform.
        """
        pass

    @abc.abstractmethod
    def jacobian_points(self, points):
        r"""
        Calculates the Jacobian of the warp with respect to the points.

        Returns
        -------
        dW_dx : (N, D, D) ndarray
            The jacobian with respect to the points
        """
        pass

    @abc.abstractmethod
    def compose(self, a):
        r"""
        Composes two transforms together::

            W(x;p) <- W(x;p) o W(x;delta_p)

        Parameters
        ----------
        a : :class:`Transform`
            Transform to be applied *FOLLOWING* self

        Returns
        --------
        transform : :class:`Transform`
            The resulting transform.
        """
        pass

    @abc.abstractproperty
    def inverse(self):
        r"""
        The inverse of the transform.

        :type: :class:`Transform`
        """
        pass

    @abc.abstractproperty
    def n_parameters(self):
        r"""
        Returns the number of parameters that determine the transform.

        :type: int
        """
        pass


class AlignmentTransform(Transform, Viewable):

    def __init__(self, source, target):
        self.source = source
        self.aligned_source = None
        self.target = target
        if source.n_dims != target.ndims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        if source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")

    @property
    def n_dims(self):
        return self.source.n_dims

    @property
    def n_points(self):
        return self.source.n_points


class Transformable(object):
    r"""
    Interface for transformable objects. When :meth:`apply` is called on
    an object, if the object has the method :meth:`_transform`,
    the method is called, passing in the transforms :meth:`apply` method.
    This allows for the object to define how it should transform itself.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _transform(self, transform):
        r"""
        Apply the transform given to the Transformable object.

        Parameters
        ----------
        transform : func
            Function that applies a transformation to the transformable object.

        Returns
        -------
        transformed : :class:`Transformable`
            The transformed object. Transformed in place.
        """
        pass
