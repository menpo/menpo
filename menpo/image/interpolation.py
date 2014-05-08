import numpy as np
from scipy.ndimage import map_coordinates


def scipy_interpolation(pixels, points_to_sample, mode='constant', order=1):
    r"""
    Interpolation utilizing SciPy's map_coordinates function.

    Parameters
    ----------
    pixels : (M, N, ..., n_channels) ndarray
        The image to be sampled from, the final axis containing channel
        information.
    points_to_sample : (n_points, n_dims) ndarray
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
    # Note that map_coordinates uses the opposite (dims, points) convention
    # to us so we transpose
    points_to_sample_t = points_to_sample.T
    for i in xrange(pixels.shape[-1]):
        sampled_pixel_values.append(map_coordinates(pixels[..., i],
                                                    points_to_sample_t,
                                                    mode=mode,
                                                    order=order))
    sampled_pixel_values = [v.reshape([-1, 1]) for v in sampled_pixel_values]
    return np.concatenate(sampled_pixel_values, axis=1)
