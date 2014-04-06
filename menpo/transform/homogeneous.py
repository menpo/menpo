import numpy as np
from menpo.transform.base import ComposableTransform


class HomogeneousTransform(ComposableTransform):
    r"""
    A simple n-dimensional homogeneous transformation.

    Adds a unit homogeneous coordinate to points, performs the dot
    product, re-normalizes by division by the homogeneous coordinate,
    and returns the result.

    Can be composed with other HomogeneousTransform's or any AffineTransform,
    so long as the dimensionality matches.

    Parameters
    ----------
    h_matrix : (n_dims + 1, n_dims + 1) ndarray
        The homogeneous matrix to be applied.
    """

    @property
    def composes_inplace_with(self):
        from menpo.transform.affine import Affine
        return HomogeneousTransform, Affine

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

    def _compose_before_inplace(self, transform):
        self.h_matrix = np.dot(transform.h_matrix, self.h_matrix)

    def _compose_after_inplace(self, transform):
        self.h_matrix = np.dot(self.h_matrix, transform.h_matrix)
