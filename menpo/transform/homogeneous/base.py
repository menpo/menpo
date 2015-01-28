import abc
import numpy as np

from menpo.base import Vectorizable
from menpo.transform.base import (Alignment, ComposableTransform,
                                  VComposable, VInvertible)


class HomogFamilyAlignment(Alignment):
    r"""
    Simple subclass of Alignment that adds the ability to create a copy of an
    alignment class without the alignment behavior.

    Note that subclasses should inherit from :map:`HomogFamilyAlignment` first
    to have the correct copy behavior.
    """
    @abc.abstractmethod
    def as_non_alignment(self):
        r"""
        Returns a copy of this transform without it's alignment nature.

        Returns
        -------
        transform : :map:`Homogeneous` but not :map:`Alignment` subclass
            A version of this transform with the same transform behavior but
            without the alignment logic.
        """

    def copy(self):
        r"""
        Generate an efficient copy of this :map:`HomogFamilyAlignment`.

        Returns
        -------
        new_transform : ``type(self)``
            A copy of this object

        """
        new = self.__class__.__new__(self.__class__)
        # Shallow copy everything except the h_matrix
        new.__dict__ = self.__dict__.copy()
        new._h_matrix = new._h_matrix.copy()
        return new

    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping source and target, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        Returns
        -------
        transform : ``type(self)``
            The inverse of this transform.
        """
        selfcopy = self.copy()
        selfcopy._h_matrix = self._h_matrix_pseudoinverse()
        selfcopy._source, selfcopy._target = selfcopy._target, selfcopy._source
        return selfcopy


class Homogeneous(ComposableTransform, Vectorizable, VComposable, VInvertible):
    r"""
    A simple n-dimensional homogeneous transformation.

    Adds a unit homogeneous coordinate to points, performs the dot
    product, re-normalizes by division by the homogeneous coordinate,
    and returns the result.

    Can be composed with another :map:`Homogeneous`, so long as the
    dimensionality matches.

    Parameters
    ----------
    h_matrix : ``(n_dims + 1, n_dims + 1)`` `ndarray`
        The homogeneous matrix defining this transform.
    copy : `bool`, optional
        If False avoid copying ``h_matrix``. Useful for performance.
    skip_checks : `bool`, optional
        If True avoid sanity checks on the ``h_matrix``. Useful for
        performance.
    """
    def __init__(self, h_matrix, copy=True, skip_checks=False):
        self._h_matrix = None
        # Delegate setting to the most specialized setter method possible
        self._set_h_matrix(h_matrix, copy=copy, skip_checks=skip_checks)

    @property
    def h_matrix_is_mutable(self):
        r"""
        ``True`` iff :meth:`set_h_matrix` is permitted on this type of
        transform.
        If this returns ``False`` calls to :meth:`set_h_matrix` will raise
        a ``NotImplementedError``.

        :type: `bool`
        """
        return True

    def from_vector(self, vector):
        """
        Build a new instance of the object from it's vectorized state.

        ``self`` is used to fill out the missing state required to
        rebuild a full object from it's standardized flattened state. This
        is the default implementation, which is which is a ``deepcopy`` of the
        object followed by a call to :meth:`from_vector_inplace()`. This method
        can be overridden for a performance benefit if desired.

        Parameters
        ----------
        vector : ``(n_parameters,)`` `ndarray`
           Flattened representation of the object.

        Returns
        -------
        transform : ``type(self)``
           An new instance of this class.
        """
        # avoid the deepcopy with an efficient copy
        self_copy = self.copy()
        self_copy.from_vector_inplace(vector)
        return self_copy

    def __str__(self):
        rep = self._transform_str() + '\n'
        rep += str(self.h_matrix)
        return rep

    def _transform_str(self):
        r"""
        A string representation explaining what this homogeneous transform
        does. Has to be implemented by base classes.

        Returns
        -------
        string : ``str``
            String representation of transform.
        """
        return 'Homogeneous'

    @classmethod
    def identity(cls, n_dims):
        return Homogeneous(np.eye(n_dims + 1))

    @property
    def h_matrix(self):
        return self._h_matrix

    def set_h_matrix(self, value, copy=True, skip_checks=False):
        r"""
        Updates ``h_matrix``, optionally performing sanity checks.

        Note that it won't always be possible to manually specify the
        ``h_matrix`` through this method, specifically if changing the
        ``h_matrix`` could change the nature of the transform. See
        :attr:`h_matrix_is_mutable` for how you can discover if the
        ``h_matrix`` is allowed to be set for a given class.

        Parameters
        ----------
        value : ndarray
            The new homogeneous matrix to set
        copy : `bool`, optional
            If False do not copy the h_matrix. Useful for performance.
        skip_checks : `bool`, optional
            If True skip checking. Useful for performance.

        Raises
        ------
        NotImplementedError
            If :attr:`h_matrix_is_mutable` returns ``False``.
        """
        if self.h_matrix_is_mutable:
            self._set_h_matrix(value, copy=copy, skip_checks=skip_checks)
        else:
            raise NotImplementedError(
                "h_matrix cannot be set on {}".format(self._transform_str()))

    def _set_h_matrix(self, value, copy=True, skip_checks=False):
        r"""
        Actually updates the h_matrix, optionally performing sanity checks.

        Called by :meth:`set_h_matrix` on classes that have
        :attr:`h_matrix_is_mutable` as ``True``.

        Every subclass should invoke this method internally when the
        h_matrix needs to be set in order to get the most sanity checking
        possible.

        Parameters
        ----------
        value : ndarray
            The new homogeneous matrix to set
        copy : `bool`, optional
            If False do not copy the h_matrix. Useful for performance.
        skip_checks : `bool`, optional
            If True skip checking. Useful for performance.
        """
        if copy:
            value = value.copy()
        self._h_matrix = value

    @property
    def n_dims(self):
        return self.h_matrix.shape[1] - 1

    @property
    def n_dims_output(self):
        # doesn't have to be a square homogeneous matrix...
        return self.h_matrix.shape[0] - 1

    def _apply(self, x, **kwargs):
        # convert to homogeneous
        h_x = np.hstack([x, np.ones([x.shape[0], 1])])
        # apply the transform
        h_y = h_x.dot(self.h_matrix.T)
        # normalize and return
        return (h_y / h_y[:, -1][:, None])[:, :-1]

    def _as_vector(self):
        return self.h_matrix.ravel()

    def from_vector_inplace(self, vector):
        self.set_h_matrix(vector.reshape(self.h_matrix.shape),
                          copy=True, skip_checks=True)

    @property
    def composes_inplace_with(self):
        r"""
        Homogeneous can swallow composition with any other Homogeneous,
        subclasses will have to override and be more specific.
        """
        return Homogeneous

    def compose_after_from_vector_inplace(self, vector):
        self.compose_after_inplace(self.from_vector(vector))

    @property
    def composes_with(self):
        r"""
        Any Homogeneous can compose with any other Homogeneous.
        """
        return Homogeneous

    # noinspection PyProtectedMember
    def _compose_before(self, t):
        r"""
        Chains an Homogeneous family transform with another transform of the
        same family, producing a new transform that is the composition of
        the two.

        .. note::

            The type of the returned transform is always the first common
            ancestor between self and transform.

            Any Alignment will be lost.

        Parameters
        ----------
        transform : :class:`Homogeneous`
            Transform to be applied **after** self

        Returns
        --------
        transform : :class:`Homogeneous`
            The resulting homogeneous transform.
        """
        # note that this overload of the basic _compose_before is just to
        # deal with the complexities of maintaining the correct class of
        # transform upon composition
        if isinstance(t, type(self)):
            # He is a subclass of me - I can swallow him.
            # What if I'm an Alignment though? Rules of composition state we
            # have to produce a non-Alignment result. Nasty, but we check
            # here to save a lot of repetition.
            if isinstance(self, HomogFamilyAlignment):
                new_self = self.as_non_alignment()
            else:
                new_self = self.copy()
            new_self._compose_before_inplace(t)
        elif isinstance(self, type(t)):
            # I am a subclass of him - he can swallow me
            new_self = t._compose_after(self)
        elif isinstance(self, Similarity) and isinstance(t, Similarity):
            # we're both in the Similarity family
            new_self = Similarity(self.h_matrix)
            new_self._compose_before_inplace(t)
        elif isinstance(self, Affine) and isinstance(t, Affine):
            # we're both in the Affine family
            new_self = Affine(self.h_matrix)
            new_self._compose_before_inplace(t)
        else:
            # at least one of us is Homogeneous
            new_self = Homogeneous(self.h_matrix)
            new_self._compose_before_inplace(t)
        return new_self

    # noinspection PyProtectedMember
    def _compose_after(self, t):
        r"""
        Chains an Homogeneous family transform with another transform of the
        same family, producing a new transform that is the composition of
        the two.

        .. note::

            The type of the returned transform is always the first common
            ancestor between self and transform.

            Any Alignment will be lost.


        Parameters
        ----------
        transform : :class:`Homogeneous`
            Transform to be applied **before** self

        Returns
        --------
        transform : :class:`Homogeneous`
            The resulting homogeneous transform.
        """
        # note that this overload of the basic _compose_after is just to
        # deal with the complexities of maintaining the correct class of
        # transform upon composition
        if isinstance(t, type(self)):
            # He is a subclass of me - I can swallow him.
            # What if I'm an Alignment though? Rules of composition state we
            # have to produce a non-Alignment result. Nasty, but we check
            # here to save a lot of repetition.
            if isinstance(self, HomogFamilyAlignment):
                new_self = self.as_non_alignment()
            else:
                new_self = self.copy()
            new_self._compose_after_inplace(t)
        elif isinstance(self, type(t)):
            # I am a subclass of him - he can swallow me
            new_self = t._compose_before(self)
        elif isinstance(self, Similarity) and isinstance(t, Similarity):
            # we're both in the Similarity family
            new_self = Similarity(self.h_matrix)
            new_self._compose_after_inplace(t)
        elif isinstance(self, Affine) and isinstance(t, Affine):
            # we're both in the Affine family
            new_self = Affine(self.h_matrix)
            new_self._compose_after_inplace(t)
        else:
            # at least one of us is Homogeneous
            new_self = Homogeneous(self.h_matrix)
            new_self._compose_after_inplace(t)
        return new_self

    def _compose_before_inplace(self, transform):
        # Compose machinery will guarantee this is only invoked in the right
        # circumstances (e.g. the types will match) so we don't need to block
        # the setting of the matrix
        self._set_h_matrix(np.dot(transform.h_matrix, self.h_matrix),
                           copy=False, skip_checks=True)

    def _compose_after_inplace(self, transform):
        # Compose machinery will guarantee this is only invoked in the right
        # circumstances (e.g. the types will match) so we don't need to block
        # the setting of the matrix
        self._set_h_matrix(np.dot(self.h_matrix, transform.h_matrix),
                           copy=False, skip_checks=True)

    @property
    def has_true_inverse(self):
        return True

    def pseudoinverse(self):
        # Skip the checks as we know inverse of a homogeneous is a homogeneous
        return self.__class__(self._h_matrix_pseudoinverse(), copy=False,
                              skip_checks=True)

    def _h_matrix_pseudoinverse(self):
        return np.linalg.inv(self.h_matrix)

from .affine import Affine
from .similarity import Similarity
