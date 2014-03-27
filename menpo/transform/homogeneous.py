import numpy as np
from .base import Transform, Composable


class HomogeneousTransform(Transform, Composable):
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

    def __init__(self, h_matrix):
        self.h_matrix = h_matrix

    @property
    def n_dims(self):
        return self.h_matrix.shape[0] - 1

    def _apply(self, x, **kwargs):
        # convert to homogeneous
        h_x = np.hstack([x, np.ones([x.shape[0], 1])])
        # apply the transform
        h_y = h_x.dot(self.h_matrix.T)
        # normalize and return
        return (h_y / h_y[:, -1][:, None])[:, :-1]

    def compose_before_inplace(self, transform):
        self.h_matrix = np.dot(transform.h_matrix, self.h_matrix)

    def compose_after_inplace(self, transform):
        self.h_matrix = np.dot(self.h_matrix, transform.h_matrix)
