import abc

from menpo.transform.base import Transform
from functools import reduce


class ComposableTransform(Transform):
    r"""
    :map:`Transform` subclass that enables native composition, such that
    the behavior of multiple :map:`Transform` s is composed together in a
    natural way.
    """

    @abc.abstractproperty
    def composes_inplace_with(self):
        r"""
        The :map:`Transform` s that this transform composes inplace
        with **natively** (i.e. no :map:`TransformChain` will be produced).

        An attempt to compose inplace against any type that is not an
        instance of this property on this class will result in an `Exception`.

        :type: :map:`Transform` or tuple of :map:`Transform` s
        """

    @property
    def composes_with(self):
        r"""
        The :map:`Transform` s that this transform composes
        with **natively** (i.e. no :map:`TransformChain` will be produced).

        If native composition is not possible, falls back to producing a
        :map:`TransformChain`.

        By default, this is the same list as :attr:`composes_inplace_with`.

        :type: :map:`Transform` or tuple of :map:`Transform` s
        """
        return self.composes_inplace_with

    def compose_before(self, transform):
        r"""
        A :map:`Transform` that represents **this** transform
        composed **before** the given transform::

            c = a.compose_before(b)
            c.apply(p) == b.apply(a.apply(p))

        ``a`` and ``b`` are left unchanged.

        An attempt is made to perform native composition, but will fall back
        to a :map:`TransformChain` as a last resort. See :attr:`composes_with`
        for a description of how the mode of composition is decided.

        Parameters
        ----------
        transform : :map:`Transform`
            Transform to be applied **after** ``self``

        Returns
        --------
        transform : :map:`Transform` or :map:`TransformChain`
            If the composition was native, a single new :map:`Transform` will
            be returned. If not, a :map:`TransformChain` is returned instead.
        """
        if isinstance(transform, self.composes_with):
            return self._compose_before(transform)
        else:
            # best we can do is a TransformChain, let Transform handle that.
            return Transform.compose_before(self, transform)

    def compose_after(self, transform):
        r"""
        A :map:`Transform` that represents **this** transform
        composed **after** the given transform::

            c = a.compose_after(b)
            c.apply(p) == a.apply(b.apply(p))

        ``a`` and ``b`` are left unchanged.

        This corresponds to the usual mathematical formalism for the compose
        operator, `o`.

        An attempt is made to perform native composition, but will fall back
        to a :map:`TransformChain` as a last resort. See :attr:`composes_with`
        for a description of how the mode of composition is decided.

        Parameters
        ----------
        transform : :map:`Transform`
            Transform to be applied **before** ``self``

        Returns
        --------
        transform : :map:`Transform` or :map:`TransformChain`
            If the composition was native, a single new :map:`Transform` will
            be returned. If not, a :map:`TransformChain` is returned instead.
        """
        if isinstance(transform, self.composes_with):
            return self._compose_after(transform)
        else:
            # best we can do is a TransformChain, let Transform handle that.
            return Transform.compose_after(self, transform)

    def compose_before_inplace(self, transform):
        r"""
        Update ``self`` so that it represents **this** transform composed
        **before** the given transform::

            a_orig = a.copy()
            a.compose_before_inplace(b)
            a.apply(p) == b.apply(a_orig.apply(p))

        ``a`` is permanently altered to be the result of the composition.
        ``b`` is left unchanged.

        Parameters
        ----------
        transform : :attr:`composes_inplace_with`
            Transform to be applied **after** ``self``

        Raises
        ------
        ValueError
            If ``transform`` isn't an instance of :attr:`composes_inplace_with`
        """
        if isinstance(transform, self.composes_inplace_with):
            self._compose_before_inplace(transform)
        else:
            raise ValueError(
                "{} can only compose inplace with {} - not "
                "{}".format(type(self), self.composes_inplace_with,
                            type(transform)))

    def compose_after_inplace(self, transform):
        r"""
        Update ``self`` so that it represents **this** transform composed
        **after** the given transform::

            a_orig = a.copy()
            a.compose_after_inplace(b)
            a.apply(p) == a_orig.apply(b.apply(p))

        ``a`` is permanently altered to be the result of the composition. ``b``
        is left unchanged.

        Parameters
        ----------
        transform : :attr:`composes_inplace_with`
            Transform to be applied **before** ``self``

        Raises
        ------
        ValueError
            If ``transform`` isn't an instance of :attr:`composes_inplace_with`
        """
        if isinstance(transform, self.composes_inplace_with):
            self._compose_after_inplace(transform)
        else:
            raise ValueError(
                "{} can only compose inplace with {} - not "
                "{}".format(type(self), self.composes_inplace_with,
                            type(transform)))

    def _compose_before(self, transform):
        r"""
        Naive implementation of composition, ``self.copy()`` and then
        :meth:``compose_before_inplace``. Apply this transform **first**.

        Parameters
        ----------
        transform : :map:`ComposableTransform`
            Transform to be applied **after** ``self``

        Returns
        --------
        transform : :map:`ComposableTransform`
            The resulting transform.
        """
        # naive approach - copy followed by the inplace operation
        self_copy = self.copy()
        self_copy._compose_before_inplace(transform)
        return self_copy

    def _compose_after(self, transform):
        r"""
        Naive implementation of composition, ``self.copy()`` and then
        :meth:``compose_after_inplace``. Apply this transform **second**.

        Parameters
        ----------
        transform : :map:`ComposableTransform`
            Transform to be applied **before** ``self``

        Returns
        --------
        transform : :map:`ComposableTransform`
            The resulting transform.
        """
        # naive approach - copy followed by the inplace operation
        self_copy = self.copy()
        self_copy._compose_after_inplace(transform)
        return self_copy

    @abc.abstractmethod
    def _compose_before_inplace(self, transform):
        r"""
        Specialised inplace composition. This should be overridden to
        provide specific cases of composition as defined in
        :attr:`composes_inplace_with`.

        Parameters
        ----------
        transform : :attr:`composes_inplace_with`
            Transform to be applied **after** ``self``
        """

    @abc.abstractmethod
    def _compose_after_inplace(self, transform):
        r"""
        Specialised inplace composition. This should be overridden to
        provide specific cases of composition as defined in
        :attr:`composes_inplace_with`.

        Parameters
        ----------
        transform : :attr:`composes_inplace_with`
            Transform to be applied **before** ``self``
        """


class VComposable(object):
    r"""
    Mix-in for :map:`Vectorizable` :map:`ComposableTransform` s.

    Use this mix-in with :map:`ComposableTransform` if the
    :map:`ComposableTransform` in question is :map:`Vectorizable` as this adds
    :meth:`from_vector` variants to the :map:`ComposableTransform` interface.
    These can be tuned for performance.
    """

    @abc.abstractmethod
    def compose_after_from_vector_inplace(self, vector):
        r"""
        Specialised inplace composition with a vector. This should be
        overridden to provide specific cases of composition whereby the current
        state of the transform can be derived purely from the provided vector.

        Parameters
        ----------
        vector : ``(n_parameters,)`` ndarray
            Vector to update the transform state with.
        """


class TransformChain(ComposableTransform):
    r"""
    A chain of transforms that can be efficiently applied one after the
    other.

    This class is the natural product of composition. Note that objects may
    know how to compose themselves more efficiently - such objects
    implement the :map:`ComposableTransform` or :map:`VComposable` interfaces.

    Parameters
    ----------
    transforms : `list` of :map:`Transform`
        The list of transforms to be applied. Note that the first transform
        will be applied first - the result of which is fed into the second
        transform and so on until the chain is exhausted.
    """

    def __init__(self, transforms):
        # TODO Should TransformChain copy on input?
        self.transforms = transforms

    def _apply(self, x, **kwargs):
        r"""
        Applies each of the transforms to the array ``x``, in order.

        Parameters
        ----------
        x : ``(n_points, n_dims)`` `ndarray`
            The array to transform.

        Returns
        -------
        transformed : ``(n_points, n_dims_output)`` `ndarray`
            Transformed array having passed through the chain of transforms.
        """
        return reduce(lambda x_i, tr: tr._apply(x_i), self.transforms, x)

    @property
    def composes_inplace_with(self):
        r"""
        The :map:`Transform` s that this transform composes inplace
        with **natively** (i.e. no :map:`TransformChain` will be produced).

        An attempt to compose inplace against any type that is not an
        instance of this property on this class will result in an `Exception`.

        :type: :map:`Transform` or tuple of :map:`Transform` s
        """
        return Transform

    def _compose_before_inplace(self, transform):
        r"""
        Specialised inplace composition. In this case we merely keep a list
        of :map:`Transform` s to apply in order.

        Parameters
        ----------
        transform : :map:`ComposableTransform`
            Transform to be applied **after** ``self``
        """
        self.transforms.append(transform)

    def _compose_after_inplace(self, transform):
        r"""
        Specialised inplace composition. In this case we merely keep a list
        of :map:`Transform`s to apply in order.

        Parameters
        ----------
        transform : :map:`ComposableTransform`
            Transform to be applied **before** ``self``
        """
        self.transforms.insert(0, transform)
