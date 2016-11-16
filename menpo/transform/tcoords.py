import numpy as np
from menpo.transform import Homogeneous, Scale


def tcoords_to_image_coords(image_shape):
    r"""
    Returns a :map:`Homogeneous` transform that converts [0,1]
    texture coordinates (tcoords) used on :map:`TexturedTriMesh`
    instances to image coordinates, which behave just like image landmarks
    do.

    The operations that are performed are:

      - Flipping the origin from bottom-left to top-left
      - Permuting the axis so that  st (or uv) -> yx
      - Scaling the tcoords by the image shape (denormalising them). Note that
        (1, 1) has to map to the highest pixel value, which is actually
        (h - 1, w - 1) due to Menpo being 0-based with image operations.

    Parameters
    ----------
    image_shape : `tuple`
        The shape of the texture that the tcoords index in to.

    Returns
    -------
    :map:`Homogeneous`
        A transform that, when applied to texture coordinates, converts them
        to image coordinates.
    """
    # flip the 'y' st 1 -> 0 and 0 -> 1, moving the axis to upper left
    invert_unit_y = Homogeneous(np.array(
        [[1., 0., 0.],
         [0., -1., 1.],
         [0., 0., 1.]]))

    # flip axis 0 and axis 1 so indexing is as expected
    flip_xy_yx = Homogeneous(np.array(
        [[0., 1., 0.],
         [1., 0., 0.],
         [0., 0., 1.]]))

    return (invert_unit_y
            .compose_before(flip_xy_yx)
            .compose_before(Scale(np.array(image_shape) - 1)))


def image_coords_to_tcoords(image_shape):
    r"""
    Returns a :map:`Homogeneous` transform that converts image coordinates
    (e.g. image landmarks) to texture coordinates (tcoords) as used on
    :map:`TexturedTriMesh` instances.

    The operations that are performed are:

      - Normalizing by image shape (e.g. converting to [0, 1]). Note that
        (1, 1) has to map to the highest pixel value, which is actually
        (h - 1, w - 1) due to Menpo being 0-based with image operations.
      - Permuting the axis so that yx -> st (or uv)
      - Flipping the origin from top-left to bottom-left

    Parameters
    ----------
    image_shape : `tuple`
        The shape of the texture that the image coordinates are on.

    Returns
    -------
    :map:`Homogeneous`
        A transform that, when applied to image coordinates, converts them
        to texture coordinates (tcoords).
    """
    return tcoords_to_image_coords(image_shape).pseudoinverse()
