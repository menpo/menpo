import numpy as np

from menpo.transform import Translation, Scale
from menpo.transform.base import Transform


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
    points: :class:`PointCloud`
        The 3D points that will be used to find the optimum unwrapping position.

    Returns
    -------

    transform: TransformChain
        A TransformChain which performs the optimal translation and unwrapping.

    """
    from menpo.misctools.circlefit import circle_fit
    from menpo.transform import Translation
    # find the optimum centre to unwrap
    xy = points.points[:, [0, 2]]  # just in the x-z plane
    centre, radius = circle_fit(xy)
    # convert the 2D circle data into the 3D space
    translation = np.array([centre[0], 0, centre[1]])
    centring_transform = Translation(-translation)
    unwrap = CylindricalUnwrap(radius)
    return centring_transform.compose_before(unwrap)


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

    @property
    def n_dims(self):
        # can operate on any dimensional shape
        return None

    @property
    def n_dims_output(self):
        # output is just n
        return self.n

    def _apply(self, x, **kwargs):
        return x[:, :self.n].copy()


class AppendNDims(Transform):
    r"""
    Adds n dims to a shape

    Parameters
    ----------

    n : int
        The number of dimensions to add

    value: float, optional
        The value that the extra dims should be given

        Default: 0
    """
    def __init__(self, n, value=0):
        self.n = n
        self.value = value

    @property
    def n_dims(self):
        # can operate on any dimensional shape
        return None

    @property
    def n_dims_output(self):
        # output is unknown (we don't in general know the input!)
        return None

    def _apply(self, x, **kwargs):
        return np.hstack([x, np.ones([x.shape[0], self.n]) * self.value]).copy()


def clip_space_transform(points, boundary_proportion=0.1):
    r"""
    Produces an SimilarityTransform which fits 3D points into the OpenGL
    clipping space ([-1, 1], [-1, 1], [-1, 1]), or 2D points into the 2D
    OpenGL clipping space ([-1, 1], [-1, 1]).

    Parameters
    ----------

    points: :class:`PointCloud`
        The points that should be adjusted.

    boundary_proprtion: float 0-1, optional
        An amount by which the boundary is relaxed (so the points are not
        right against the edge)

        Default: 0.1 (10% reduction in tight crop size)

    Returns
    -------
    :class:`SimilarityTransform
        The similarity transform that creates this mapping
    """
    centering = Translation(points.centre_of_bounds).pseudoinverse
    scale = Scale(points.range() / 2)
    b_scale = Scale(1 - boundary_proportion, n_dims=points.n_dims)
    return centering.compose_before(scale.pseudoinverse).compose_before(b_scale)
