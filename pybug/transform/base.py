import abc
from copy import deepcopy
from pybug.base import Vectorizable
from pybug.visualize.base import Viewable


class Transform(Vectorizable):
    r"""
    An abstract representation of any N-dimensional transform.
    Provides a unified interface to apply the transform (:meth:`apply`). All
    transforms are vectorizable.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, source=None, target=None):
        self._source = source
        self.aligned_source = None
        self._target = target

    @classmethod
    def align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target.

        Parameters
        ----------

        source: :class:`pybug.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`pybug.shape.PointCloud`
            The target pointcloud instance used in the alignment
        """
        if source.n_dims != target.ndims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        if source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")
        return cls._align(source, target, **kwargs)

    @classmethod
    def _align(cls, source, target, **kwargs):
        pass

    @abc.abstractproperty
    def n_dims(self):
        r"""`
        The dimensionality of the transform.

        :type: int
        """
        pass

    @abc.abstractproperty
    def n_parameters(self):
        r"""
        The number of parameters that determine the transform.

        :type: int
        """
        pass

    @property
    def is_alignment_transform(self):
        return self.source is not None and self.target is not None

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        r"""
        Updates this alignment transform to point to a new target.
        """
        if not self.is_alignment_transform:
            raise NotImplementedError("Cannot update target for Transforms "
                                      "not built with the align constructor")
        else:
            if value.n_dims != self.target.n_dims:
                raise ValueError(
                    "The current target is {}D, the new target is {}D - new "
                    "target has to have the same dimensionality as the "
                    "old".format(self.target.n_dims, value.n_dims))
            elif value.n_points != self.target.n_points:
                raise ValueError(
                    "The current target has {} points, the new target has {} "
                    "- new target has to have the same number of points as the"
                    " old".format(self.target.n_points, value.n_points))
            else:
                old_target = self._target
                self._target = value
                self._update_from_target(old_target)

    @abc.abstractmethod
    def _update_from_target(self, old_target):
        r"""
        Updates this alignment transform based on the newly set target.
        """
        pass

    def from_target(self, target):
        r"""
        Returns a new instance of this alignment transform with the source
        unchanged but the target set to a newly provided target.

        Parameters
        ----------

        target: :class:`pybug.shape.PointCloud`
            The new target that should be used in this align transform.
        """
        new_transform = deepcopy(self)
        new_transform.target = target
        return new_transform

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


class AlignmentTransform(Transform):
    r"""
    :class:`Transform`s that are defined in terms of a source and target.
    """

    def __init__(self, source, target):
        super(AlignmentTransform, self).__init__(source=source, target=target)
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
