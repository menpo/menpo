import abc
from copy import deepcopy


class Transform(object):
    r"""
    An abstract representation of any spatial transform.
    Provides a unified interface to apply the transform with
    :meth:`apply_inplace` and :meth:`apply`. All Transforms support basic
    composition to form :class:`TransformChain`.

    For native composition, see the :class:`ComposableTransform` subclass and
    the :class:`VComposition` mix-in.
    For inversion, see the :class:`Invertable` and :class:`VInvertable` mix-ins.
    For alignment, see the :class:`Alignment` mix in.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def n_dims(self):
        r"""
        The dimensionality of the data the transform operates on.
        None if the transform is not dimension specific.

        :type: int or None
        """
        return None

    @property
    def n_dims_output(self):
        r"""
        The output of the data from the transform. None if the output
        of the transform is not dimension specific.

        :type: int or None
        """
        # most Transforms don't change the dimensionality of their input.
        return self.n_dims

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

    def apply_inplace(self, x, **kwargs):
        r"""
        Applies this transform to ``x``. If ``x`` is :class:`Transformable`,
        ``x`` will be handed this transform object to transform itself
        inplace. If not, ``x`` is assumed to be a numpy array. The
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

    def compose_before(self, transform):
        r"""
        c = a.compose_before(b)
        c.apply(p) == b.apply(a.apply(p))

        a and b are left unchanged.

        Parameters
        ----------
        transform : :class:`Transform`
            Transform to be applied **after** self

        Returns
        --------
        transform : :class:`TransformChain`
            The resulting transform chain.
        """
        return TransformChain([self, transform])

    def compose_after(self, transform):
        r"""
        c = a.compose_after(b)
        c.apply(p) == a.apply(b.apply(p))

        a and b are left unchanged.

        This corresponds to the usual mathematical formalism for the compose
        operator, `o`.

        Parameters
        ----------
        transform : :class:`Transform`
            Transform to be applied **before** self

        Returns
        --------
        transform : :class:`TransformChain`
            The resulting transform chain.
        """
        return TransformChain([transform, self])


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
        Apply the transform given in a non destructive manner - returning the
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


from .alignment import Alignment
from .composable import TransformChain, ComposableTransform, VComposable
from .invertable import Invertible, VInvertible
