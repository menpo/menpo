import abc
from copy import deepcopy
from menpo.transform.base import Transform


class ComposableTransform(Transform):
    r"""
    Transform subclass that enables native composition, such that
    behavior of multiple Transforms is compounded together in a natural
    way.

    """

    @abc.abstractproperty
    def composes_inplace_with(self):
        r"""Class or tuple of Classes that this transform composes
        inplace against natively.

        An attempt to compose inplace against any type that is not an
        instance of this property on this class will result in an Exception.
        """
        pass

    @property
    def composes_with(self):
        r"""Class or tuple of Classes that this transform composes against
        natively.
        If native composition is not possible, falls back to producing a
        :class:`TransformChain`.

        By default, this is the same list as composes_inplace_with.
        """
        return self.composes_inplace_with

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
        if isinstance(transform, self.composes_with):
            return self._compose_before(transform)
        else:
            # best we can do is a TransformChain, let Transform handle that.
            return Transform.compose_before(self, transform)

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
        if isinstance(transform, self.composes_with):
            return self._compose_after(transform)
        else:
            # best we can do is a TransformChain, let Transform handle that.
            return Transform.compose_after(self, transform)

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

        Raises
        ------
        ValueError: If this transform cannot be composed inplace with the
        provided transform
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
        -------
        transform : self
            self, updated to the result of the composition

        Raises
        ------
        ValueError: If this transform cannot be composed inplace with the
        provided transform
        """
        if isinstance(transform, self.composes_inplace_with):
            self._compose_after_inplace(transform)
        else:
            raise ValueError(
                "{} can only compose inplace with {} - not "
                "{}".format(type(self), self.composes_inplace_with,
                            type(transform)))

    def _compose_before(self, transform):
        # naive approach - deepcopy followed by the inplace operation
        new_transform = deepcopy(self)
        new_transform._compose_before_inplace(transform)
        return new_transform

    def _compose_after(self, transform):
        # naive approach - deepcopy followed by the inplace operation
        new_transform = deepcopy(self)
        new_transform._compose_after_inplace(transform)
        return new_transform

    @abc.abstractmethod
    def _compose_before_inplace(self, transform):
        pass

    @abc.abstractmethod
    def _compose_after_inplace(self, transform):
        pass


class VComposable(object):

    @abc.abstractmethod
    def compose_after_from_vector_inplace(self, vector):
        pass


class TransformChain(ComposableTransform):
    r"""
    A chain of transforms that can be efficiently applied one after the other.

    This class is the natural product of composition. Note that objects may
    know how to compose themselves more efficiently - such objects are
    implementing the Compose or VCompose interface.

    Parameters
    ----------
    transforms : list of :class:`Transform`
        The list of transforms to be applied. Note that the first transform
        will be applied first - the result of which is fed into the second
        transform and so on until the chain is exhausted.

    """
    def __init__(self, transforms):
        # TODO for now we don't copy, important to come back and evaluate
        self.transforms = transforms

    def _apply(self, x, **kwargs):
        r"""
        Applies each of the transforms to the array ``x``, in order.

        Parameters
        ----------
        x : (N, D) ndarray

        Returns
        -------
        transformed : (N, D) ndarray
            Transformed array having passed through the chain of transforms.
        """
        return reduce(lambda x_i, tr: tr._apply(x_i), self.transforms, x)

    @property
    def composes_inplace_with(self):
        return Transform

    def _compose_before_inplace(self, transform):
        self.transforms.append(transform)

    def _compose_after_inplace(self, transform):
        self.transforms.insert(0, transform)
