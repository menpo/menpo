import abc
from copy import deepcopy
from menpo.transform.base import Transform

__author__ = 'jab08'


class ComposableTransform(Transform):
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
    :class:`menpo.shape.PointCloud` p around it's mean, and then translate
    it some place else. It would be nice to be able to do something like

        t = Translation(-p.centre)  # translate to centre
        s = Scale(2.0)  # rescale
        move = Translate([10, 0 ,0]) # budge along the x axis

        t.compose(s).compose(-t).compose(move)

    in Menpo, this functionality is provided by the compose_before() family
    of methods.

        (a.compose_before(b)).apply(x) == b.apply(a.apply(x))

    within each family there are two methods, some of which may provide
    performance benefits for certain situations. they are

        compose_x(transform)
        compose_x_inplace(transform)

    where x = {after, before}

    See specific subclasses for more information about the performance of
    these methods.
    """

    @abc.abstractproperty
    def composes_inplace_with(self):
        r"""Iterable of classes that this transform composes against natively.
        If native composition is not possible, falls back to producing a
        :class:`TransformChain`
        """
        pass

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
        if isinstance(transform, self.composes_inplace_with):
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
        if isinstance(transform, self.composes_inplace_with):
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
                "{}".format(type(self), self.composes_inplace_with, type(transform)))

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
                "{}".format(type(self), self.composes_inplace_with, type(transform)))

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

    def _compose_before_inplace(self, transform):
        self.transforms.append(transform)

    def _compose_after_inplace(self, transform):
        self.transforms.insert(0, transform)

    @property
    def composes_inplace_with(self):
        return Transform

    def __init__(self, transforms):

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


class VComposableTransform(ComposableTransform):
    r"""
    Transform Mixin for Vectorizable composable Transforms.

    Prefer this Mixin over Composable if the Transform in question is
    Vectorizable, as this adds from_vector variants to the Composable
    interface. These can be tuned for performance, and are for instance
    needed by some of the machinery of AAMs.
    """

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

    def _compose_after_inplace(self, transform):
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