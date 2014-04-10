import abc
import copy
import numpy as np

from menpo.base import Vectorizable
from menpo.transform.base import (Alignment, ComposableTransform,
                                  VComposable, VInvertible)


class HomogFamilyAlignment(Alignment):
    r"""
    Simple subclass of Alignment that adds the ability to create a copy of an
    alignment class without the alignment behavior.
    """
    @abc.abstractmethod
    def copy_without_alignment(self):
        pass


class Homogeneous(ComposableTransform, Vectorizable, VComposable, VInvertible):
    r"""
    A simple n-dimensional homogeneous transformation.

    Adds a unit homogeneous coordinate to points, performs the dot
    product, re-normalizes by division by the homogeneous coordinate,
    and returns the result.

    Can be composed with another Homogeneous, so long as the dimensionality
    matches.

    Parameters
    ----------
    h_matrix : (n_dims + 1, n_dims + 1) ndarray
        The homogeneous matrix to be applied.

    """
    def __init__(self, h_matrix):
        self._h_matrix = h_matrix.copy()

    @classmethod
    def identity(cls, n_dims):
        return Homogeneous(np.eye(n_dims + 1))

    @property
    def h_matrix(self):
        return self._h_matrix

    def set_h_matrix(self, value):
        # TODO add verification logic for homogeneous here
        self._h_matrix = value.copy()

    @property
    def n_dims(self):
        return self.h_matrix.shape[0] - 1

    @property
    def n_dims_output(self):
        # doesn't have to be a square homogeneous matrix...
        return self.h_matrix.shape[1] - 1

    def _apply(self, x, **kwargs):
        # convert to homogeneous
        h_x = np.hstack([x, np.ones([x.shape[0], 1])])
        # apply the transform
        h_y = h_x.dot(self.h_matrix.T)
        # normalize and return
        return (h_y / h_y[:, -1][:, None])[:, :-1]

    def as_vector(self):
        return self.h_matrix.flatten()

    def from_vector_inplace(self, vector):
        self.set_h_matrix(vector.reshape(self.h_matrix.shape))

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
        from .affine import Affine
        from .similarity import Similarity
        if isinstance(t, type(self)):
            # He is a subclass of me - I can swallow him.
            # What if I'm an Alignment though? Rules of composition state we
            # have to produce a non-Alignment result. Nasty, but we check
            # here to save a lot of repetition.
            if isinstance(self, HomogFamilyAlignment):
                new_self = self.copy_without_alignment()
            else:
                new_self = copy.deepcopy(self)
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
        from .affine import Affine
        from .similarity import Similarity
        if isinstance(t, type(self)):
            # He is a subclass of me - I can swallow him.
            # What if I'm an Alignment though? Rules of composition state we
            # have to produce a non-Alignment result. Nasty, but we check
            # here to save a lot of repetition.
            if isinstance(self, HomogFamilyAlignment):
                new_self = self.copy_without_alignment()
            else:
                new_self = copy.deepcopy(self)
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
        # Force the Homogeneous variant. compose machinery will guarantee
        # this is only invoked in the right circumstances (e.g. the types
        # will match so we don't need to block the setting of the matrix)
        Homogeneous.set_h_matrix(self, np.dot(transform.h_matrix,
                                              self.h_matrix))

    def _compose_after_inplace(self, transform):
        # Force the Homogeneous variant. compose machinery will guarantee
        # this is only invoked in the right circumstances (e.g. the types
        # will match so we don't need to block the setting of the matrix)
        Homogeneous.set_h_matrix(self, np.dot(self.h_matrix,
                                              transform.h_matrix))

    def has_true_inverse(self):
        return True

    def _build_pseudoinverse(self):
        return Homogeneous(np.linalg.inv(self.h_matrix))
