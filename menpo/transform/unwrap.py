import numpy as np
from menpo.math import radial_fit
from .base import Transform
from .homogeneous import Translation


class CylindricalUnwrap(Transform):
    r"""
    Maps 3D points :math:`(x, y, z)` into cylindrical coordinates
    :math:`(x', y', z')`

    .. math::

        x' &\leftarrow r \theta \\
        y' &\leftarrow y \\
        z' &\leftarrow d

    where

    .. math::

        d &= \sqrt{x^2 + y^2} - r \\
        \theta &= \arctan{\left(\frac{x}{z}\right)}

    The cylinder is oriented such that it's axial vector is ``[0, 1, 0]``
    and it's centre is at the origin. Discontinuity in :math:`\theta` values
    occurs at ``y-z`` plane for *negative* ``z`` values (i.e. the interesting
    information you are wanting to preserve in the unwrapping better have
    positive ``z`` values).

    Parameters
    ----------

    radius : `float`
        The distance of the unwrapping from the axis.

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
    Returns a :map:`TransformChain` of
    [:map:`Translation`, :map:`CylindricalUnwrap`]
    which optimally cylindrically unwraps the points provided. This is done by:

    #. Find an optimal :map:`Translation` to centre the points in ``x-z`` plane
    #. Use :map:`circle_fit` to find the optimal radius for fitting the points
    #. Calculate a :map:`CylindricalUnwrap` using the optimal radius
    #. Return a composition of the two.

    Parameters
    ----------
    points : :map:`PointCloud`
        The 3D points that will be used to find the optimum unwrapping position

    Returns
    -------

    transform: :map:`TransformChain`
        A :map:`TransformChain` which performs the optimal translation and
        unwrapping.

    """
    # find the optimum centre to unwrap
    xy = points.points[:, [0, 2]]  # just in the x-z plane
    centre, radius = radial_fit(xy)
    # convert the 2D circle data into the 3D space
    translation = np.array([centre[0], 0, centre[1]])
    centring_transform = Translation(-translation)
    unwrap = CylindricalUnwrap(radius)
    return centring_transform.compose_before(unwrap)
