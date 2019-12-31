import numpy as np

from .interpolation import scipy_interpolation


def _centered_patch(patch_shape):
    r"""
    Helper method for generating the relative coordinates of a centered
    patch with shape (prod(patch_shape), 2). First generate a centered patch
    where the (0, 0) coordinate is in the center of the patch.
    Therefore, the (0, 0) pixel value is actually negative. The reshape to be in
    the form of sampling locations.

    Parameters
    ----------
    patch_shape : (R, C)

    Returns
    -------
    sampling_locations : ``(prod(patch_shape), 2)`` ``ndarray``
        The locations to sample for the given patch shape
    """
    assert len(patch_shape) == 2, 'Only 2D images are supported'
    half_pixel = (np.array([patch_shape]) % 2) / 2
    patch = np.meshgrid(
        np.linspace(-patch_shape[0] / 2, patch_shape[0] / 2, num=patch_shape[0],
                    dtype=np.float, endpoint=False),
        np.linspace(-patch_shape[1] / 2, patch_shape[1] / 2, num=patch_shape[1],
                    dtype=np.float, endpoint=False),
        indexing='ij'
    )
    patch = np.stack(patch, axis=2).reshape([-1, 2])
    return np.require(patch, requirements=['C']) + half_pixel


def extract_patches_by_sampling(pixels, patch_centers, patch_shape,
                                offsets=None, order=0, mode='constant',
                                cval=0.0):
    r"""
    Extract a set of patches from the given pixels. Given a set of patch centers
    and a patch size, patches are extracted from within the pixels, centered
    on the given coordinates. Note that only 2D images are currently supported.
    Note that this is much less efficient than slicing.

    Parameters
    ----------
    pixels : ``(n_channels, height, width)`` `ndarray``
        Pixels to extract patches from
    patch_centers : ``(n_patches, 2)`` `ndarray``
        The centers to extract patches around.
    patch_shape : ``(1, 2)`` `tuple` or `ndarray`, optional
        The size of the patch to extract
    offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
        Offsets to sample from within a patch. So ``(0, 0)`` is the
        centre of the patch (no offset) and ``(1, 0)`` would be sampling the
        patch from 1 pixel up the first axis away from the centre.
        If ``None``, then no offsets are applied.
    order : `int`, optional
        The order of interpolation. The order has to be in the range [0,5].
        See warp_to_shape for more information.
    mode : ``{constant, nearest, reflect, wrap}``, optional
        Points outside the boundaries of the input are filled according
        to the given mode.
    cval : `float`, optional
        Used in conjunction with mode ``constant``, the value outside
        the image boundaries.

    Returns
    -------
    patches : ``(n_center, n_offset, n_channels, patch_shape)`` `ndarray
        The extracted patches including any offsets if they were requested.

    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    if pixels.ndim != 3:
        raise ValueError('Only 2D images are supported but '
                         'found {}'.format(pixels.shape))

    n_points = patch_centers.shape[0]
    n_offsets = 1

    patch = _centered_patch(patch_shape)
    points_to_sample = patch[:, None, :] + patch_centers
    if offsets is not None:
        n_offsets = offsets.shape[0]
        points_to_sample = points_to_sample[:, :, None, :] + offsets
    points_to_sample = points_to_sample.reshape([-1, 2])

    patches = scipy_interpolation(pixels, points_to_sample,
                                  order=order, mode=mode, cval=cval)
    patches = patches.reshape(3, patch_shape[0], patch_shape[1],
                              n_points, n_offsets)
    patches = np.transpose(patches, [3, 4, 0, 1, 2])
    return np.require(patches, requirements=['C'])


def extract_patches_with_slice(pixels, patch_centers, patch_shape,
                               offsets=None, cval=0.):
    r"""
    Extract a set of patches from the given pixels. Given a set of patch centers
    and a patch size, patches are extracted from within the pixels, centered
    on the given coordinates. Note that only 2D images are currently supported.
    This is equivalent to sampling with ``order=0`` and ``mode='constant'``.

    Parameters
    ----------
    pixels : ``(n_channels, height, width)`` `ndarray``
        Pixels to extract patches from
    patch_centers : ``(n_patches, 2)`` `ndarray``
        The centers to extract patches around.
    patch_shape : ``(1, 2)`` `tuple` or `ndarray`, optional
        The size of the patch to extract
    offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
        Offsets to sample from within a patch. So ``(0, 0)`` is the
        centre of the patch (no offset) and ``(1, 0)`` would be sampling the
        patch from 1 pixel up the first axis away from the centre.
        If ``None``, then no offsets are applied.
    cval : `float`, optional
        The value outside the image boundaries.

    Returns
    -------
    patches : ``(n_center, n_offset, n_channels, patch_shape)`` `ndarray
        The extracted patches including any offsets if they were requested.

    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    if pixels.ndim != 3:
        raise ValueError('Only 2D images are supported but '
                         'found {}'.format(pixels.shape))

    n_offsets = offsets.shape[0] if offsets is not None else 1
    half_r, half_c = (patch_shape[0] / 2, patch_shape[1] / 2)
    # These are the offsets to the corners of the patches (hence the negative)
    corners = np.array([[-half_r, -half_c], [half_r, half_c]])
    if offsets is None:
        offsets = np.zeros([1, 2])

    patches = np.full([patch_centers.shape[0],
                       n_offsets,
                       pixels.shape[0],
                       patch_shape[0],
                       patch_shape[1]],
                      fill_value=cval,
                      dtype=pixels.dtype)
    # This is equivalent to nearest neighbour sampling per offset
    half_pixel = (np.array([patch_shape]) % 2) / 2
    patch_centers = patch_centers + half_pixel
    bounds = np.round(patch_centers[:, None, None, :] +
                      offsets[:, None, :] +
                      corners).astype(int)
    # Limit the points to exist inside the image boundaries
    pixel_bounds = np.clip(bounds, [0, 0], [pixels.shape[1:]])
    # Then compute the regions inside the patches that need to be filled
    patch_bounds = pixel_bounds - bounds

    # pixel_bounds will have shape (n_points, n_offsets, 2, 2)
    # Loop over the patches
    for i, (pix_off, patch_off) in enumerate(zip(pixel_bounds, patch_bounds)):
        # Loop over the offsets
        for j, (pix_b, patch_b) in enumerate(zip(pix_off, patch_off)):
            pix_slice = (slice(pix_b[0, 0], pix_b[1, 0]),
                         slice(pix_b[0, 1], pix_b[1, 1]))
            patch_slice = (slice(patch_b[0, 0], patch_shape[0] + patch_b[1, 0]),
                           slice(patch_b[0, 1], patch_shape[1] + patch_b[1, 1]))

            # Set only the pixels that were inside the image bounds
            patches[i, j, :, patch_slice[0], patch_slice[1]] = \
                pixels[:, pix_slice[0], pix_slice[1]]

    return patches


def set_patches(patches, pixels, patch_centers, offset, offset_index):
    r"""
    Set the values of a group of patches into the correct regions of a copy
    of this image. Given an array of patches and a set of patch centers,
    the patches' values are copied in the regions of the image that are
    centred on the coordinates of the given centers.

    The patches argument can have any of the two formats that are returned
    from the `extract_patches()` and `extract_patches_around_landmarks()`
    methods. Specifically it can be:

        1. ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
        2. `list` of ``n_center * n_offset`` :map:`Image` objects

    Currently only 2D images are supported.

    Parameters
    ----------
    patches : `ndarray` or `list`
        The values of the patches.
        A ``(n_center, n_offset, self.n_channels, patch_shape)`` `ndarray`
    pixels : ``(n_channels, height, width)`` `ndarray``
        Pixel array to replace the patches within
    patch_centers : :map:`PointCloud`
        The centers to set the patches around.
    offset : `list` or `tuple` or ``(1, 2)`` `ndarray`
        The offset to apply on the patch centers within the image.
    offset_index : `int`
        The offset index within the provided `patches` argument, thus the
        index of the second dimension from which to sample.

    Raises
    ------
    ValueError
        If pixels array is not 2D
    """
    if pixels.ndim != 3:
        raise ValueError('Only 2D images are supported but '
                         'found {}'.format(pixels.shape))

    patch_shape = patches.shape[-2:]
    # the [L]ow offset is the floor of half the patch shape
    l_r, l_c = (int(patch_shape[0] // 2), int(patch_shape[1] // 2))
    # the [H]igh offset needs to be one pixel larger if the original
    # patch was odd
    h_r, h_c = (int(l_r + patch_shape[0] % 2), int(l_c + patch_shape[1] % 2))
    for patches_with_offsets, point in zip(patches, patch_centers):
        patch = patches_with_offsets[offset_index]
        p = point + offset[0]
        p_r = int(p[0])
        p_c = int(p[1])
        pixels[:, p_r - l_r:p_r + h_r, p_c - l_c:p_c + h_c] = patch
