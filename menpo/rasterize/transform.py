import numpy as np
from menpo.transform.base import Transform


# TODO this could be a general useful transform (higher than Affine)
class LinearHTransform(Transform):

    def __init__(self, h_matrix):
        self.h_matrix = h_matrix

    def _apply(self, x, **kwargs):
        # convert to homogeneous
        h_x = np.hstack([x, np.ones([x.shape[0], 1])])
        # apply the transform
        h_y = h_x.dot(self.h_matrix.T)
        # normalize and return
        return (h_y / h_y[:, -1][:, None])[:, :-1]


class CylindricalUnwrapTransform(Transform):
    r"""
    Unwraps 3D points into cylindrical coordinates:
    x -> radius * theta
    y -> z
    z -> depth

    The cylinder is oriented st. it's axial vector is [0, 1, 0]
    and it's centre is at the origin. discontinuity in theta values
    occurs at y-z plane for NEGATIVE z values (i.e. the interesting
    information you are wanting to unwrap better have positive z values).

    radius - the distance of the unwrapping from the axis.
    z -  the distance along the axis of the cylinder (maps onto the y
         coordinate exactly)
    theta - the angular distance around the cylinder, in radians. Note
         that theta itself is not outputted but theta * radius, preserving
         distances.

    depth - is the displacement away from the radius along the radial vector.
    """
    def __init__(self, radius):
        self.radius = radius

    def _apply(self, x, **kwargs):
        cy_coords = np.zeros_like(x)
        depth = np.sqrt(x[:, 0]**2 + x[:, 2]**2) - self.radius
        theta = np.arctan2(x[:, 0], x[:, 2])
        z = x[:, 1]
        cy_coords[:, 0] = theta * self.radius
        cy_coords[:, 1] = z
        cy_coords[:, 2] = depth
        return cy_coords


class ExtractNDims(Transform):
    r"""
    Extracts out the first n dimensions of a shape.

    Parameters
    ----------

    n : int
        The number of dimensions to extract
    """
    def __init__(self, n):
        self.n = n

    def _apply(self, x, **kwargs):
        return x[:, :self.n].copy()


class AddNDims(Transform):
    r"""
    Adds n dims to a shape
    """
    def __init__(self, n, value=0):
        self.n = n
        self.value = value

    def _apply(self, x, **kwargs):
        return np.hstack([x, np.ones([x.shape[0], self.n]) * self.value]).copy()
