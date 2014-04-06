import copy
import numpy as np

from menpo.base import Vectorizable
from menpo.transform.base import ComposableTransform, VInvertible


# Vectorizable, VComposableTransform, VInvertible
class Homogeneous(ComposableTransform, Vectorizable, VInvertible):
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
        self.h_matrix = h_matrix

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

    @classmethod
    def identity(cls, n_dims):
        return Homogeneous(np.eye(n_dims + 1))

    def as_vector(self):
        return self.h_matrix.flatten()

    def from_vector_inplace(self, vector):
        self.h_matrix = vector.reshape(self.h_matrix.shape)

    @property
    def composes_inplace_with(self):
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
            # He is a subclass of me - I can swallow him
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
            # He is a subclass of me - I can swallow him
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
        self.h_matrix = np.dot(transform.h_matrix, self.h_matrix)

    def _compose_after_inplace(self, transform):
        self.h_matrix = np.dot(self.h_matrix, transform.h_matrix)

    def has_true_inverse(self):
        return True

    def _build_pseudoinverse(self):
        return Homogeneous(np.linalg.inv(self.h_matrix))
