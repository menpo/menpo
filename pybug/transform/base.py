import abc
from copy import deepcopy
import numpy as np
from pybug.base import Vectorizable
from pybug.visualize import AlignmentViewer2d
from pybug.visualize.base import Viewable


class Transform(Vectorizable):
    r"""
    An abstract representation of any N-dimensional transform.
    Provides a unified interface to apply the transform (
    :meth:`apply_inplace`, :meth:`apply`). All
    transforms are Vectorizable. Transform's know how to take their own
    jacobians, be composed, and construct their pseduoinverse.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractproperty
    def n_dims(self):
        r"""`
        The dimensionality of the transform.

        :type: int
        """
        pass

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
        Calculates the Jacobian at the points provided.

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
    def _build_pseudoinverse(self):
        r"""
        Returns this transform's inverse if it has one. if not,
        the pseduoinverse is given.

        This method is called by the pseudoinverse property and must be
        overridden.


        Returns
        -------
        pseudoinverse: type(self)
        """
        pass

    @abc.abstractproperty
    def has_true_inverse(self):
        r"""
        True if the pseudoinverse is an exact inverse.

        :type: Boolean
        """
        pass

    @property
    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping source and target, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        :type: :class:`Transform`
        """
        return self._build_pseudoinverse()

    def apply_inplace(self, x, **kwargs):
        r"""
        Applies this transform to ``x``. If ``x`` is :class:`Transformable`,
        ``x`` will be handed this transform object to transform itself
        inplace. If not, ``x`` is assumed to be a numpy array. The
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
            x._transform_inplace(transform)
        except AttributeError:
            x[...] = self._apply(x, **kwargs)

    def apply(self, x, **kwargs):
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
            return x._transform(transform)
        except AttributeError:
            return self._apply(x, **kwargs)

    def pseudoinverse_vector(self, vector):
        r"""
        The vectorized pseudoinverse of a provided vector instance.

        Syntactic sugar for

        self.from_vector(vector).pseudoinverse.as_vector()

        Can be much faster than the explict call as object creation can be
        entirely avoided in some cases.

        Parameters
        ----------
        vector :  (P,) ndarray
            A vectorized version of self

        Returns
        -------
        pseudoinverse_vector : (N,) ndarray
            The pseudoinverse of the vector provided
        """
        return self.from_vector(vector).pseudoinverse.as_vector()


class Composable(object):
    r"""
    Mixin for Transform objects that can be composed together, such that
    behavior of multiple Transforms is compounded together in some way.

    There are two useful forms of composition. Firstly, the mathematical
    composition symbol `o` has the definition

        let a(x) and b(x) be two transforms on x.
        (a o b)(x) == a(b(x))

    This functionality is provided by the compose_after() family of methods.

        (a.compose_after(b)).apply(x) == a.apply(b.apply(x))

    Equally useful is an inversion the order of composition - so that over
    time a large chains of transforms can be built up that do a useful job,
    and composing on this chain adds another transform to the end (after all
    other preceding transforms have been performed).

    For instance, let's say we want to rescale a
    :class:`pybug.shape.PointCloud` p around it's mean, and then translate
    it some place else. It would be nice to be able to do something like

        t = Translation(-p.centre)  # translate to centre
        s = Scale(2.0)  # rescale
        move = Translate([10, 0 ,0]) # budge along the x axis

        t.compose(s).compose(-t).compose(move)

    in PyBug, this functionality is provided by the compose_before() family
    of methods.

        (a.compose_before(b)).apply(x) == b.apply(a.apply(x))

    within each family there are three methods, some of which may provide
    performance benefits for certain situations. they are

        compose_x(transform)
        compose_x_inplace(transform)
        compose_x_from_vector_inplace(vectorized_transform)

    where x = {after, before}

    See specific subclasses for more information about the performance of
    these methods.
    """
    __metaclass__ = abc.ABCMeta

    def compose_before(self, transform):
        r"""
        c = a.compose_before(b)
        c.apply(p) == b.apply(a.apply(p))

        a and b are left unchanged.

        Parameters
        ----------
        transform : :class:`Composable`
            Transform to be applied **after** self

        Returns
        --------
        transform : :class:`Composable`
            The resulting transform.
        """
        # naive approach  - deepcopy followed by the inplace operation
        new_transform = deepcopy(self)
        return new_transform.compose_before_inplace(transform)

    @abc.abstractmethod
    def compose_before_inplace(self, transform):
        r"""
        a_orig = deepcopy(a)
        a.compose_before_inplace(b)
        a.apply(p) == b.apply(a_orig.apply(p))

        a is permanently altered to be the result of the composition. b is
        left unchanged.

        Parameters
        ----------
        transform : :class:`Composable`
            Transform to be applied **after** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        pass

    def compose_before_from_vector_inplace(self, vector):
        r"""
        a_orig = deepcopy(a)
        a.compose_before_from_vector_inplace(b_vec)
        b = self.from_vector(b_vec)
        a.apply(p) == b.apply(a.apply(p))

        a is permanently altered to be the result of the composition. b_vec
        is left unchanged.

        Parameters
        ----------
        vector : (N,) ndarray
            Vectorized transform to be applied **after** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        # naive approach - use the vector to build an object,
        # then compose_before_inplace
        return self.compose_before_inplace(self.from_vector(vector))

    def compose_after(self, transform):
        r"""
        c = a.compose_after(b)
        c.apply(p) == a.apply(b.apply(p))

        a and b are left unchanged.

        This corresponds to the usual mathematical formalism for the compose
        operator, `o`.

        Parameters
        ----------
        transform : :class:`Composable`
            Transform to be applied **before** self

        Returns
        --------
        transform : :class:`Composable`
            The resulting transform.
        """
        # naive approach - just flip the object order and compose_before
        return transform.compose_before(self)

    def compose_after_inplace(self, transform):
        r"""
        a_orig = deepcopy(a)
        a.compose_after_inplace(b)
        a.apply(p) == a_orig.apply(b.apply(p))

        a is permanently altered to be the result of the composition. b is
        left unchanged.

        Parameters
        ----------
        transform : :class:`Composable`
            Transform to be applied **before** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        # naive approach - update self to be equal to transform and
        # compose_before_from_vector_inplace
        self_vector = self.as_vector().copy()
        self.update_from_vector(transform.as_vector())
        return self.compose_before_from_vector_inplace(self_vector)

    def compose_after_from_vector_inplace(self, vector):
        r"""
        a_orig = deepcopy(a)
        a.compose_after_from_vector_inplace(b_vec)
        b = self.from_vector(b_vec)
        a.apply(p) == a_orig.apply(b.apply(p))

        a is permanently altered to be the result of the composition. b_vec
        is left unchanged.

        Parameters
        ----------
        vector : (N,) ndarray
            Vectorized transform to be applied **before** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        # naive approach - use the vector to build an object,
        # then compose_after_inplace
        return self.compose_after_inplace(self.from_vector(vector))


class AlignableTransform(Transform):
    r"""
    Abstract interface for all transform's that can be constructed from an
    optimisation aligning a source PointCloud to a target PointCloud.
    Construction from the align class method enables certain features of hte
    class, like the from_target() and update_from_target() method. If the
    instance is just constructed with it's regular constructor, it functions
    as a normal Transform - attempting to call alignment methods listed here
    will simply yield an Exception.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        Transform.__init__(self)
        self._target = None
        self._source = None

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

        alignment_transform: :class:`pybug.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        pass

    @abc.abstractmethod
    def _target_setter(self, new_target):
        r"""
        Updates this alignment transform based on the new target.

        It is the responsibility of this method to leave the object in the
        updated state, including setting new_target to self._target as
        appropriate. Note that this method is called by the target setter,
        so this behavior must be respected.
        """
        pass

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

        alignment_transform: :class:`pybug.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        cls._verify_source_and_target(source, target)
        return cls._align(source, target, **kwargs)

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
                self._target_setter(value)

    @property
    def aligned_source(self):
        if not self.is_alignment_transform:
            raise ValueError("This is not an alignment transform")
        else:
            return self.apply(self.source)

    @property
    def alignment_error(self):
        r"""
        The Frobenius Norm of the difference between the target and
        the aligned source.

        :type: float
        """
        return np.linalg.norm(self.target.points - self.aligned_source.points)

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


class PureAlignmentTransform(AlignableTransform, Viewable):
    r"""
    :class:`AlignableTransform`s that are solely defined in terms of a source
    and target alignment.

    All transforms include support for alignments - all have a source and
    target property the alignment constructor, and methods like
    from_target(). However, for most transforms this is an optional
    interface - if the alignment constructor is not used, is_alignment is
    false, and all alignment methods will fail.

    This class is for transforms that solely make sense as alignments. It
    simplifies the interface down, so that :class:`PureAlignmentTransform`
    subclasses only have to override :meth:`_target_setter()`
    to satisfy the AlignableTransform interface.
    """

    def __init__(self, source, target):
        AlignableTransform.__init__(self)
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
        View the PureAlignmentTransform. This plots the source points and
        vectors that represent the shift from source to target.

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
        PureAlignmentTransform's we know that align == __init__. To save
        repetition we share the align method here.

        Parameters
        ----------

        source: :class:`pybug.shape.PointCloud`
            The source pointcloud instance used in the alignment

        target: :class:`pybug.shape.PointCloud`
            The target pointcloud instance used in the alignment

        Returns
        -------

        alignment_transform: :class:`pybug.transform.AlignableTransform`
            A Transform object that is_alignment.
        """
        return cls(source, target, **kwargs)


class Transformable(object):
    r"""
    Interface for transformable objects. When :meth:`apply_inplace` is called
    on an object, if the object has the method :meth:`_transform_inplace`,
    the method is called, passing in the transforms :meth:`apply_inplace`
    method.
    This allows for the object to define how it should transform itself.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _transform_inplace(self, transform):
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

    def _transform(self, transform):
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
        copy_of_self._transform_inplace(transform)
        return copy_of_self
