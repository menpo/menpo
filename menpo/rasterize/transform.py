import numpy as np

from menpo.transform import Translation, Scale, NonUniformScale, Homogeneous
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


def model_to_clip_transform(points, xy_scale=0.9, z_scale=0.1):
    r"""
    Produces an Affine Transform which centres and scales 3D points to fit
    into the OpenGL clipping space ([-1, 1], [-1, 1], [-1, 1]). This can be
    used to construct an appropriate projection matrix for use in an
    orthographic Rasterizer.

    Parameters
    ----------

    points: :class:`PointCloud`
        The points that should be adjusted.

    xy_scale: float 0-1, optional
        Amount by which the boundary is relaxed so the points are not
        right against the edge. A value of 1 means the extremities of the
        point cloud will be mapped onto [-1, 1] [-1, 1] exactly (no boarder)
        A value of 0.5 means the points will be mapped into the range
        [-0.5, 0.5].

        Default: 0.9 (map to [-0.9, 0.9])

    z_scale: float 0-1, optional
        Scale factor by which the z-dimension is squeezed. A value of 1
        means the z-range of the points will be mapped to exactly fit in
        [-1, 1]. A scale of 0.1 means the z-range is compressed to fit in the
        range [-0.1, 0.1].

    Returns
    -------
    :class:`Affine`
        The affine transform that creates this mapping
    """
    # 1. Centre the points on the origin
    centering = Translation(points.centre_of_bounds).pseudoinverse
    # 2. Scale the points to exactly fit the boundaries
    scale = Scale(points.range() / 2.0)
    # 3. Apply the relaxations requested
    b_scale = NonUniformScale([xy_scale, xy_scale, z_scale])
    return centering.compose_before(scale.pseudoinverse).compose_before(b_scale)



def clip_to_image_transform(width, height):
    r"""
    Affine transform that converts 3D clip space coordinates into 2D image
    space coordinates. Note that the z axis of the clip space coordinates is
    ignored.

    Parameters
    ----------

    width: int
        The width of the image

    height: int
        The height of the image

    Returns
    -------

    HomogeneousTransform
        A homogeneous transform that moves clip space coordinates into image
        space.
    """
    from menpo.transform import Homogeneous, Translation, Scale
    # 1. Remove the z axis from the clip space
    rem_z = Homogeneous(np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]]))
    # 2. invert the y direction (up becomes down)
    invert_y = Scale([1, -1])
    # 3. [-1, 1] [-1, 1] -> [0, 2] [0, 2]
    t = Translation([1, 1])
    # 4. [0, 2] [0, 2] -> [0, 1] [0, 1]
    unit_scale = Scale(0.5, n_dims=2)
    # 5. [0, 1] [0, 1] -> [0, w] [0, h]
    im_scale = Scale([width, height])
    # 6. [0, w] [0, h] -> [0, h] [0, w]
    xy_yx = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]], dtype=np.float))
    # reduce the full transform chain to a single affine matrix
    transforms = [rem_z, invert_y, t, unit_scale, im_scale, xy_yx]
    return reduce(lambda a, b: a.compose_before(b), transforms)
