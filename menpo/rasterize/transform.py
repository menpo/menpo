from functools import reduce
import numpy as np
from menpo.transform import Homogeneous, Translation, Scale, NonUniformScale


def dims_3to2():
    r"""

    Returns
     ------
    :map`Homogeneous`
        :map`Homogeneous` that strips off the 3D axis of a 3D shape
        leaving just the first two axes.

    """
    return Homogeneous(np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]]))


def dims_2to3(x=0):
    r"""
    Return a :map`Homogeneous` that adds on a 3rd axis to a 2D shape.

    Parameters
    ----------
    x : `float`, optional
        The value that will be assigned to the new third dimension

    Returns
     ------
    :map`Homogeneous`
        :map`Homogeneous` that adds on a 3rd axis to a 2D shape.

    """
    return Homogeneous(np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, x],
                                 [0, 0, 1]]))


def model_to_clip_transform(points, xy_scale=0.9, z_scale=0.3):
    r"""
    Produces an Affine Transform which centres and scales 3D points to fit
    into the OpenGL clipping space ([-1, 1], [-1, 1], [1, 1-]). This can be
    used to construct an appropriate projection matrix for use in an
    orthographic Rasterizer. Note that the z-axis is flipped as is default in
    OpenGL - as a result this transform converts the right handed coordinate
    input into a left hand one.

    Parameters
    ----------

    points: :map:`PointCloud`
        The points that should be adjusted.

    xy_scale: `float` 0-1, optional
        Amount by which the boundary is relaxed so the points are not
        right against the edge. A value of 1 means the extremities of the
        point cloud will be mapped onto [-1, 1] [-1, 1] exactly (no boarder)
        A value of 0.5 means the points will be mapped into the range
        [-0.5, 0.5].

        Default: 0.9 (map to [-0.9, 0.9])

    z_scale: float 0-1, optional
        Scale factor by which the z-dimension is squeezed. A value of 1
        means the z-range of the points will be mapped to exactly fit in
        [1, -1]. A scale of 0.1 means the z-range is compressed to fit in the
        range [0.1, -0.1].

    Returns
    -------
    :map:`Affine`
        The affine transform that creates this mapping
    """
    # 1. Centre the points on the origin
    center = Translation(points.centre_of_bounds).pseudoinverse
    # 2. Scale the points to exactly fit the boundaries
    scale = Scale(points.range() / 2.0)
    # 3. Apply the relaxations requested - note the flip in the z axis!!
    # This is because OpenGL by default evaluates depth as bigger number ==
    # further away. Thus not only do we need to get to clip space [-1, 1] in
    # all dims) but we must invert the z axis so depth buffering is correctly
    # applied.
    b_scale = NonUniformScale([xy_scale, xy_scale, -z_scale])
    return center.compose_before(scale.pseudoinverse).compose_before(b_scale)


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
    # 1. Remove the z axis from the clip space
    rem_z = dims_3to2()
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
