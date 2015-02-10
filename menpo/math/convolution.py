# log_gabor filter and  __frequency_butterworth_filter are derived from Matlab
# scripts written by Peter Kovesi. We maintain his copyright notice below.
#
# Copyright (c) 1999 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# http://www.csse.uwa.edu.au/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.

import numpy as np


def __adjusted_meshgrid(shape):
    """
    Creates an adjusted meshgrid that accounts for odd image sizes. Linearly
    interpolates the values. This meshgrid assumes 'ij' indexing - which is
    due to the 1st dimension of an image being the y-dimension.

    Parameters
    ----------
    shape: tuple
        Size of meshgrid, (M, N, ...). The dimensionality should not be
        swapped due to using images. Therefore, for a 2D image, the expected
        tuple is `(HEIGHT, WIDTH)`.

    Returns
    -------
    meshgrid : list of (M, N, ...) ndarrays
        The meshgrid over each dimension given by the shape.

    """
    adjust_range = []
    for dim in shape:
        adjust_range.append(np.linspace(-0.5, 0.5, dim))

    return np.meshgrid(*adjust_range, indexing='ij')


def __frequency_butterworth_filter(shape, cutoff, order):
    r"""
    Builds an N-D butterworth filter

        ..math::

            f = \frac{1.0}{1.0 + (w / cutoff)^{2n}}

    The frequency origin of the returned filter is at the corners.

    Parameters
    ----------
    shape : tuple
        The size of the filter (M, N, ...)
    cutoff : double
        Cutoff frequency of the filter in the range `[0, 0.5]`
    order : positive int
        Order of the filter. The higher it is the sharper the transition

    Returns
    -------
    butterworth_filter : (M, N, ...) ndarray
        The butterworth filter for the given parameters. Will be the same
        shape as was requested.
    """
    # Dimension-free sum of squares
    grid = __adjusted_meshgrid(shape)
    grid_sq = [g ** 2 for g in grid]
    grid_sq = sum(grid_sq)

    radius = np.sqrt(grid_sq)
    return np.fft.ifftshift(1.0 / ((radius / cutoff) ** (2 * order) + 1.0))


# TODO: merge the 2D and 3D versions if possible
def log_gabor(image, **kwargs):
    r"""
    Creates a log-gabor filter bank, including smoothing the images via a
    low-pass filter at the edges.

    To create a 2D filter bank, simply specify the number of phi
    orientations (orientations in the xy-plane).

    To create a 3D filter bank, you must specify both the number of
    phi (azimuth) and theta (elevation) orientations.

    This algorithm is directly derived from work by Peter Kovesi.

    Parameters
    ----------
    image : ``(M, N, ...)`` `ndarray`
        Image to be convolved
    num_scales : `int`, optional
        Number of wavelet scales.

        ========== ==
        Default 2D 4
        Default 3D 4
        ========== ==
    num_phi_orientations : `int`, optional
        Number of filter orientations in the xy-plane

        ========== ==
        Default 2D 6
        Default 3D 6
        ========== ==
    num_theta_orientations : `int`, optional
        **Only required for 3D**. Number of filter orientations in the z-plane

        ========== ==
        Default 2D N/A
        Default 3D 4
        ========== ==
    min_wavelength : `int`, optional
        Wavelength of smallest scale filter.

        ========== ==
        Default 2D 3
        Default 3D 3
        ========== ==
    scaling_constant : `int`, optional
        Scaling factor between successive filters.

        ========== ==
        Default 2D 2
        Default 3D 2
        ========== ==
    center_sigma : `float`, optional
        Ratio of the standard deviation of the Gaussian describing the Log
        Gabor filter's transfer function in the frequency domain to the filter
        centre frequency.

        ========== ==
        Default 2D 0.65
        Default 3D 0.65
        ========== ==
    d_phi_sigma : `float`, optional
        Angular bandwidth in xy-plane

        ========== ==
        Default 2D 1.3
        Default 3D 1.5
        ========== ==
    d_theta_sigma : `float`, optional
        **Only required for 3D**. Angular bandwidth in z-plane

        ========== ==
        Default 2D N/A
        Default 3D 1.5
        ========== ==

    Returns
    -------
    complex_conv : ``(num_scales, num_orientations, image.shape)`` `ndarray`
        Complex valued convolution results. The real part is the
        result of convolving with the even symmetric filter, the
        imaginary part is the result from convolution with the
        odd symmetric filter.
    bandpass : ``(num_scales, image.shape)`` `ndarray`
        Bandpass images corresponding to each scale `s`
    S : ``(image.shape,)`` `ndarray`
        Convolved image

    Examples
    --------
    Return the magnitude of the convolution over the image at
    scale `s` and orientation `o`

    ::

        np.abs(complex_conv[s, o, :, :])

    Return the phase angles

    ::

        np.angle(complex_conv[s, o, :, :])

    References
    ----------
    .. [1] D. J. Field, "Relations Between the Statistics of Natural Images
        and the Response Properties of Cortical Cells",
        Journal of The Optical Society of America A, Vol 4, No. 12,
        December 1987. pp 2379-2394
    """
    if len(image.shape) == 2:    # 2D filter
        return __log_gabor_2d(image, **kwargs)
    elif len(image.shape) == 3:  # 3D filter
        return __log_gabor_3d(image, **kwargs)
    else:
        raise ValueError("Image must be either 2D or 3D")


def __log_gabor_3d(image, num_scales=4, num_phi_orientations=6,
                   num_theta_orientations=4, min_wavelength=3,
                   scaling_constant=2, center_sigma=0.65, d_theta_sigma=1.5,
                   d_phi_sigma=1.5):
    # Pre-compute sigma values
    theta_sigma = np.pi / num_theta_orientations / d_theta_sigma
    phi_sigma = (2 * np.pi) / num_phi_orientations / d_phi_sigma

    # Allocate space for return structures
    bandpass = np.empty([num_scales, image.shape[0], image.shape[1],
                         image.shape[2]], dtype=np.complex)
    log_gabor = np.empty([num_scales, image.shape[0], image.shape[1],
                          image.shape[2]])
    S = np.zeros(image.shape)
    complex_conv = np.empty([num_scales, num_theta_orientations,
                             num_phi_orientations, image.shape[0],
                             image.shape[1], image.shape[2]], dtype=np.complex)
    tmp_complex_conv = np.empty([num_scales, image.shape[0], image.shape[1],
                                 image.shape[2]], dtype=np.complex)

    # Pre-compute fourier values
    image_fft = np.fft.fftn(image)

    axis0, axis1, axis2 = __adjusted_meshgrid(image.shape)

    radius = np.sqrt(axis0 ** 2 + axis1 ** 2 + axis2 ** 2)
    theta = np.arctan2(axis0, axis1)
    # TODO: Is adding the mean REALLY a good idea?
    m_ab = np.abs(np.mean(radius))
    phi = np.arccos(axis2 / (radius + m_ab))

    radius = np.fft.ifftshift(radius)
    radius[0, 0, 0] = 1.0
    theta = np.fft.ifftshift(theta)
    phi = np.fft.ifftshift(phi)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Compute the lowpass filter
    butterworth_filter = __frequency_butterworth_filter(image.shape, 0.45, 15)

    # Compute radial component of filter
    for s in range(num_scales):
        wavelength = min_wavelength * scaling_constant ** s
        fo = 1.0 / wavelength

        l = np.exp((-np.log(radius / fo) ** 2) /
                   (2.0 * np.log(center_sigma) ** 2))
        l = l * butterworth_filter
        l[0, 0, 0] = 0.0

        log_gabor[s, :, :, :] = l
        bandpass[s, :, :, :] = np.fft.ifft2(image_fft * l)

    # Computer angular component of filter
    for e in range(num_theta_orientations):
        # Pre-compute filter data specific to this orientation
        elevation_angle = e * np.pi / num_theta_orientations

        d_theta_sin = (sin_theta * np.cos(elevation_angle) -
                       cos_theta * np.sin(elevation_angle))
        d_theta_cos = (cos_theta * np.cos(elevation_angle) +
                       sin_theta * np.sin(elevation_angle))
        d_theta = np.abs(np.arctan2(d_theta_sin, d_theta_cos))

        for a in range(num_phi_orientations):
            azimuth_angle = a * 2 * np.pi / num_phi_orientations
            d_phi_sin = (sin_phi * np.cos(azimuth_angle) -
                         cos_phi * np.sin(azimuth_angle))
            d_phi_cos = (cos_phi * np.cos(azimuth_angle) +
                         sin_phi * np.sin(azimuth_angle))
            d_phi = np.abs(np.arctan2(d_phi_sin, d_phi_cos))

            phi_spread = (-d_phi ** 2) / (2 * phi_sigma ** 2)
            theta_spread = (-d_theta ** 2) / (2 * theta_sigma ** 2)
            spread = np.exp(phi_spread + theta_spread)

            # For each scale, multiply by the angular spread
            for s in range(0, num_scales):
                filter_bank = log_gabor[s] * spread

                shifted_filter = np.fft.fftshift(filter_bank)
                S += shifted_filter * np.conjugate(shifted_filter)

                tmp_complex_conv[s, :, :] = np.fft.ifft2(image_fft *
                                                         filter_bank)

            complex_conv[:, e, a, :, :] = tmp_complex_conv[None, None, ...]

    # TODO: Do we need to flip S as in the 2D version?
    return complex_conv, bandpass, S


def __log_gabor_2d(image, num_scales=4, num_orientations=6,
                   min_wavelength=3, scaling_constant=2, center_sigma=0.65,
                   d_phi_sigma=1.3):
    # Allocate space for return structures
    bandpass = np.empty([num_scales, image.shape[0], image.shape[1]],
                        dtype=np.complex)
    log_gabor = np.empty([num_scales, image.shape[0], image.shape[1]])
    S = np.zeros(image.shape)
    complex_conv = np.empty([num_scales, num_orientations, image.shape[0],
                             image.shape[1]], dtype=np.complex)
    tmp_complex_conv = np.empty([num_scales, image.shape[0], image.shape[1]],
                                dtype=np.complex)

    # Pre-compute phi sigma
    phi_sigma = np.pi / num_orientations / d_phi_sigma

    # Pre-compute fourier values
    image_fft = np.fft.fft2(image)

    axis0, axis1 = __adjusted_meshgrid(image.shape)

    radius = np.sqrt(axis0 ** 2 + axis1 ** 2)
    phi = np.arctan2(axis0, axis1)

    radius = np.fft.ifftshift(radius)
    radius[0][0] = 1.0
    phi = np.fft.ifftshift(phi)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Compute the lowpass filter
    butterworth_filter = __frequency_butterworth_filter(image.shape, 0.45, 15)

    # Compute radial component of filter
    for s in range(num_scales):
        wavelength = min_wavelength * scaling_constant ** s
        fo = 1.0 / wavelength

        l = np.exp((-(np.log(radius / fo)) ** 2) /
                   (2.0 * np.log(center_sigma) ** 2))
        l = l * butterworth_filter
        l[0][0] = 0.0

        log_gabor[s, :, :] = l
        bandpass[s, :, :] = np.fft.ifft2(image_fft * l)

    # Computer angular component of filter
    for o in range(num_orientations):
        # Pre-compute filter data specific to this orientation
        filter_angle = o * np.pi / num_orientations

        ds = (sin_phi * np.cos(filter_angle) -
              cos_phi * np.sin(filter_angle))
        dc = (cos_phi * np.cos(filter_angle) +
              sin_phi * np.sin(filter_angle))

        d_phi = np.abs(np.arctan2(ds, dc))

        # Calculate the standard deviation of the angular Gaussian
        # function used to construct filters in the freq. plane.
        spread = np.exp((-d_phi ** 2.0) / (2.0 * phi_sigma ** 2))

        # For each scale, multiply by the angular spread
        for s in range(0, num_scales):
            filter_bank = log_gabor[s] * spread

            shifted_filter = np.fft.fftshift(filter_bank)
            S += shifted_filter * np.conjugate(shifted_filter)

            tmp_complex_conv[s, :, :] = np.fft.ifft2(image_fft * filter_bank)

        complex_conv[:, o, :, :] = tmp_complex_conv[None, ...]

    # TODO: Why is this done??
    return complex_conv, bandpass, np.flipud(S)
