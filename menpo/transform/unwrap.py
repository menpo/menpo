import numpy as np
from menpo.math import circle_fit
from .base import Transform
from .homogeneous import Translation


class CylindricalUnwrap(Transform):
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


def optimal_cylindrical_unwrap(points):
    r"""
    Returns a TransformChain of [Translation, CylindricalUnwrap] which
    optimally cylindrically unwraps the points provided. This is done by:

    a. Finding the optimal Translation to centre the points provided (in the
    x-z plane)

    b. Calculating a CylindricalUnwrap using the optimal radius from a

    c. Returning the chain of the two.

    Parameters
    ----------
    points: :map:`PointCloud`
        The 3D points that will be used to find the optimum unwrapping position

    Returns
    -------

    transform: :map:`TransformChain`
        A :map:`TransformChain` which performs the optimal translation and
        unwrapping.

    """
    # find the optimum centre to unwrap
    xy = points.points[:, [0, 2]]  # just in the x-z plane
    centre, radius = circle_fit(xy)
    # convert the 2D circle data into the 3D space
    translation = np.array([centre[0], 0, centre[1]])
    centring_transform = Translation(-translation)
    unwrap = CylindricalUnwrap(radius)
    return centring_transform.compose_before(unwrap), radius
