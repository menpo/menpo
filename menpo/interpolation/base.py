import numpy as np
from scipy.ndimage import map_coordinates
from menpo.interpolation.cinterp import interp2


def c_interpolation(ndimage, points_to_sample, mode='bilinear'):
    r"""
    C-based interpolator that was designed to be identical when
    used in both Python and Matlab.

    Parameters
    ----------
    ndimage : (M, N, ..., C) ndarray
        The image that is to be sampled from. The final axis channels.
    points_to_sample: (K, n_points) ndarray
        The points which should be sampled from pixels
    mode : {'bilinear', 'bicubic', 'nearest'}, optional
        The type of interpolation to be carried out.

        Default: bilinear

    Returns
    -------
    sampled_image : ndarray
        The pixel information sampled at each of the points.
    """
    return interp2(ndimage, points_to_sample[0, :], points_to_sample[1, :],
                   mode=mode)


def scipy_interpolation(pixels, points_to_sample, mode='constant', order=1):
    r"""
    C-based interpolator that was designed to be identical when
    used in both Python and Matlab.

    Parameters
    ----------
    ndimage : (M, N, ..., C) ndarray
        The image that is to be sampled from. The final axis channels.
    points_to_sample: (K, n_points) ndarray
        The points which should be sampled from pixels
    mode : {'constant', 'nearest', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according to the
        given mode.

        Default: 'constant' (0)
    order : int, optional
        The order of the spline interpolation. The order has to be in the
        range 0-5.

        Default: 1

    Returns
    -------
    sampled_image : ndarray
        The pixel information sampled at each of the points.
    """
    sampled_pixel_values = []
    # Loop over every channel in image - we know last axis is always channels
    for i in xrange(pixels.shape[-1]):
        sampled_pixel_values.append(map_coordinates(pixels[..., i],
                                                    points_to_sample,
                                                    mode=mode,
                                                    order=order))
    sampled_pixel_values = [v.reshape([-1, 1]) for v in sampled_pixel_values]
    return np.concatenate(sampled_pixel_values, axis=1)
