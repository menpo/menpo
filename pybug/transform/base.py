import abc
from copy import deepcopy
from pybug.base import Vectorizable
from pybug.visualize import AlignmentViewer2d
from pybug.visualize.base import Viewable


class Alignment(object):
    r"""
    Abstract interface for alignements.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._target = None
        self._source = None

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

        Returns
        -------

        alignment_transform: :class:`pybug.transform.Transform`
            A Transform object that is_alignment.
        """
        cls._verify_source_and_target(source, target)
        return cls._align(source, target, **kwargs)

    @classmethod
    def _align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target.

        Parameters
        ----------

        source: :class:`pybug.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`pybug.shape.PointCloud`
            The target pointcloud instance used in the alignment

        This is called automatically by align once verification of source and
        target is performed.

        Returns
        -------

        alignment_transform: :class:`pybug.transform.Transform`
            A Transform object that is_alignment.
        """
        pass

    @staticmethod
    def _verify_source_and_target(source, target):
        if source.n_dims != target.n_dims:
            raise ValueError("Source and target must have the same "
                             "dimensionality")
        elif source.n_points != target.n_points:
            raise ValueError("Source and target must have the same number of"
                             " points")
        else:
            return True

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
    def _update_from_target(self, new_target):
        r"""
        Updates this alignment transform based on the new target.

        It is the responsibility of this method to leave the object in the
        updated state, including setting new_target to self._target as
        appropriate. Note that this method is called by the target setter,
        so this behavior must be respected.
        """
        pass

    def from_target(self, target):
        r"""
        Returns a new instance of this alignment transform with the source
        unchanged but the target set to a newly provided target.

        This default implementation simply deep copy's the current
        transform, and then changes the target in place. If there is a more
        efficient way for transforms to perform this operation, they can
        just subclass this method.

        Parameters
        ----------

        target: :class:`pybug.shape.PointCloud`
            The new target that should be used in this align transform.
        """
        new_transform = deepcopy(self)
        # If this method is overridden in a subclass verification of target
        # will have to be called manually (right now it is called in the
        # target setter here).
        new_transform.target = target
        return new_transform


class Transform(Alignment, Vectorizable):
    r"""
    An abstract representation of any N-dimensional transform.
    Provides a unified interface to apply the transform (:meth:`apply`). All
    transforms are Vectorizable.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        Alignment.__init__(self)

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

    def apply(self, x, **kwargs):
        r"""
        Applies this transform to ``x``. If ``x`` is :class:`Transformable`,
        ``x`` will be handed this transform object to transform itself
        destructively. If not, ``x`` is assumed to be a numpy array. The
        transformation will be non destructive, returning the transformed
        version. Any ``kwargs`` will be passed to the specific transform
        :meth:`_apply` methods.

        If you wish to apply a Transform to a Transformable in a nondestructive
        manor, use the nondestructive keyword

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

    def apply_nondestructive(self, x, **kwargs):
        r"""
        Applies this transform to ``x``. If ``x`` is :class:`Transformable`,
        ``x`` will be handed this transform object to transform itself
        non-destructively (a transformed copy of the object will be
        returned).
        If not, ``x`` is assumed to be a numpy array. The transformation
        will be non destructive, returning the transformed version. Any
        ``kwargs`` will be passed to the specific transform
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
            return x._transform_nondestructive(transform)
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


class AlignmentTransform(Transform, Viewable):
    r"""
    :class:`Transform`s that are solely defined in terms of a source and
    target.

    All transforms include support for alignments - all have a source and
    target property the alignment constructor, and methods like
    from_target(). However, for most transforms this is an optional
    interface - if the alignment constructor is not used, is_alignment is
    false, and all alignment methods will fail. This class is for transforms
    that solely make sense as alignments. It just simplifies the interface down
    slightly, to remove code repetition.
    """

    def __init__(self, source, target):
        Transform.__init__(self)
        if self._verify_source_and_target(source, target):
            self._source = source
            self._target = target

    @property
    def n_dims(self):
        return self.source.n_dims

    @property
    def n_points(self):
        return self.source.n_points

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        View the AlignmentTransform. This plots the source points and vectors
        that represent the shift from source to target.

        Parameters
        ----------
        image : bool, optional
            If ``True`` the vectors are plotted on top of an image

            Default: ``False``
        """
        if self.n_dims == 2:
            return AlignmentViewer2d(figure_id, new_figure, self)
        else:
            raise ValueError("Only 2D alignments can be viewed currently.")

    @classmethod
    def align(cls, source, target, **kwargs):
        r"""
        Alternative Transform constructor. Constructs a Transform by finding
        the optimal transform to align source to target. Note that for
        AlignmentTransform's we know that align == __init__. To save
        repetition we share the align method here.

        Parameters
        ----------

        source: :class:`pybug.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`pybug.shape.PointCloud`
            The target pointcloud instance used in the alignment

        Returns
        -------

        alignment_transform: :class:`pybug.transform.Transform`
            A Transform object that is_alignment.
        """
        return cls(source, target, **kwargs)


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

    def _transform_nondestructive(self, transform):
        r"""
        Apply the transform given in a non destructive manor - returning the
        transformed object and leaving this object as it was.

        Parameters
        ----------
        transform : func
            Function that applies a transformation to the transformable object.

        Returns
        -------
        transformed : :class:`Transformable`
            A copy of the object, transformed.
        """
        copy_of_self = deepcopy(self)
        # transform the copy destructively
        copy_of_self._transform(transform)
        return copy_of_self
