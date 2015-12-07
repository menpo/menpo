import warnings
import numpy as np

from menpo.base import Copyable, MenpoDeprecationWarning


class Transform(Copyable):
    r"""
    Abstract representation of any spatial transform.

    Provides a unified interface to apply the transform with
    :meth:`apply_inplace` and :meth:`apply`.

    All Transforms support basic composition to form a :map:`TransformChain`.

    There are two useful forms of composition. Firstly, the mathematical
    composition symbol `o` has the following definition::

        Let a(x) and b(x) be two transforms on x.
        (a o b)(x) == a(b(x))

    This functionality is provided by the :meth:`compose_after` family of
    methods: ::

        (a.compose_after(b)).apply(x) == a.apply(b.apply(x))

    Equally useful is an inversion the order of composition - so that over
    time a large chain of transforms can be built to do a useful job, and
    composing on this chain adds another transform to the end (after all other
    preceding transforms have been performed).

    For instance, let's say we want to rescale a :map:`PointCloud` ``p`` around
    its mean, and then translate it some place else. It would be nice to be able
    to do something like::

        t = Translation(-p.centre)  # translate to centre
        s = Scale(2.0)  # rescale
        move = Translate([10, 0 ,0])  # budge along the x axis
        t.compose(s).compose(-t).compose(move)

    In Menpo, this functionality is provided by the :meth:`compose_before()`
    family of methods::

        (a.compose_before(b)).apply(x) == b.apply(a.apply(x))

    For native composition, see the :map:`ComposableTransform` subclass and
    the :map:`VComposable` mix-in.

    For inversion, see the :map:`Invertible` and :map:`VInvertible` mix-ins.

    For alignment, see the :map:`Alignment` mix-in.
    """

    @property
    def n_dims(self):
        r"""
        The dimensionality of the data the transform operates on.

        ``None`` if the transform is not dimension specific.

        :type: `int` or ``None``
        """
        return None

    @property
    def n_dims_output(self):
        r"""
        The output of the data from the transform.

        ``None`` if the output of the transform is not dimension specific.

        :type: `int` or ``None``
        """
        # most Transforms don't change the dimensionality of their input.
        return self.n_dims

    def _apply(self, x, **kwargs):
        r"""
        Applies the transform to the array ``x``, returning the result.

        This method does the actual work of transforming the data, and is the
        one that subclasses must implement. :meth:`apply` and
        :meth:`apply_inplace` both call this method to do that actual work.

        Parameters
        ----------
        x : ``(n_points, n_dims)`` `ndarray`
            The array to be transformed.
        kwargs : `dict`
            Subclasses may need these in their ``_apply`` methods.

        Returns
        -------
        transformed : ``(n_points, n_dims_output)`` `ndarray`
            The transformed array
        """
        raise NotImplementedError()

    def apply_inplace(self, *args, **kwargs):
        r"""
        Deprecated as public supported API, use the non-mutating `apply()`
        instead.

        For internal performance-specific uses, see `_apply_inplace()`.

        """
        warnings.warn('the public API for inplace operations is deprecated '
                      'and will be removed in a future version of Menpo. '
                      'Use .apply() instead.', MenpoDeprecationWarning)
        return self._apply_inplace(*args, **kwargs)

    def _apply_inplace(self, x, **kwargs):
        r"""
        Applies this transform to a :map:`Transformable` ``x`` destructively.

        Any ``kwargs`` will be passed to the specific transform :meth:`_apply`
        method.

        Note that this is an inplace operation that should be used sparingly,
        by internal API's where creating a copy of the transformed object is
        expensive. It does not return anything, as the operation is inplace.

        Parameters
        ----------
        x : :map:`Transformable`
            The :map:`Transformable` object to be transformed.
        kwargs : `dict`
            Passed through to :meth:`_apply`.
        """

        def transform(x_):
            """
            Local closure which calls the :meth:`_apply` method with the
            `kwargs` attached.
            """
            return self._apply(x_, **kwargs)

        try:
            x._transform_inplace(transform)
        except AttributeError:
            raise ValueError('apply_inplace can only be used on Transformable'
                             ' objects.')

    def apply(self, x, batch_size=None, **kwargs):
        r"""
        Applies this transform to ``x``.

        If ``x`` is :map:`Transformable`, ``x`` will be handed this transform
        object to transform itself non-destructively (a transformed copy of the
        object will be returned).

        If not, ``x`` is assumed to be an `ndarray`. The transformation will be
        non-destructive, returning the transformed version.

        Any ``kwargs`` will be passed to the specific transform :meth:`_apply`
        method.

        Parameters
        ----------
        x : :map:`Transformable` or ``(n_points, n_dims)`` `ndarray`
            The array or object to be transformed.
        batch_size : `int`, optional
            If not ``None``, this determines how many items from the numpy
            array will be passed through the transform at a time. This is
            useful for operations that require large intermediate matrices
            to be computed.
        kwargs : `dict`
            Passed through to :meth:`_apply`.

        Returns
        -------
        transformed : ``type(x)``
            The transformed object or array
        """

        def transform(x_):
            """
            Local closure which calls the :meth:`_apply` method with the
            `kwargs` attached.
            """
            return self._apply_batched(x_, batch_size, **kwargs)

        try:
            return x._transform(transform)
        except AttributeError:
            return self._apply_batched(x, batch_size, **kwargs)

    def _apply_batched(self, x, batch_size, **kwargs):
        if batch_size is None:
            return self._apply(x, **kwargs)
        else:
            outputs = []
            n_points = x.shape[0]
            for lo_ind in range(0, n_points, batch_size):
                hi_ind = lo_ind + batch_size
                outputs.append(self._apply(x[lo_ind:hi_ind], **kwargs))
            return np.vstack(outputs)

    def compose_before(self, transform):
        r"""
        Returns a :map:`TransformChain` that represents **this** transform
        composed **before** the given transform::

            c = a.compose_before(b)
            c.apply(p) == b.apply(a.apply(p))

        ``a`` and ``b`` are left unchanged.

        Parameters
        ----------
        transform : :map:`Transform`
            Transform to be applied **after** self

        Returns
        -------
        transform : :map:`TransformChain`
            The resulting transform chain.
        """
        return TransformChain([self, transform])

    def compose_after(self, transform):
        r"""
        Returns a :map:`TransformChain` that represents **this** transform
        composed **after** the given transform::

            c = a.compose_after(b)
            c.apply(p) == a.apply(b.apply(p))

        ``a`` and ``b`` are left unchanged.

        This corresponds to the usual mathematical formalism for the compose
        operator, `o`.

        Parameters
        ----------
        transform : :map:`Transform`
            Transform to be applied **before** self

        Returns
        -------
        transform : :map:`TransformChain`
            The resulting transform chain.
        """
        return TransformChain([transform, self])


class Transformable(Copyable):
    r"""
    Interface for objects that know how to be transformed by the
    :map:`Transform` interface.

    When ``Transform.apply_inplace`` is called on an object, the
    :meth:`_transform_inplace` method is called, passing in the transforms'
    :meth:`_apply` function.

    This allows for the object to define how it should transform itself.
    """

    def _transform_inplace(self, transform):
        r"""
        Apply the given transform function to ``self`` inplace.

        Parameters
        ----------
        transform : `function`
            Function that applies a transformation to the transformable object.

        Returns
        -------
        transformed : ``type(self)``
            The transformed object, having been transformed in place.
        """
        raise NotImplementedError()

    def _transform(self, transform):
        r"""
        Apply the :map:`Transform` given in a non destructive manner -
        returning the transformed object and leaving this object as it was.

        Parameters
        ----------
        transform : `function`
            Function that applies a transformation to the transformable object.

        Returns
        -------
        transformed : ``type(self)``
            A copy of the object, transformed.
        """
        copy_of_self = self.copy()
        # transform the copy destructively
        copy_of_self._transform_inplace(transform)
        return copy_of_self


from .alignment import Alignment
from .composable import TransformChain, ComposableTransform, VComposable
from .invertible import Invertible, VInvertible
