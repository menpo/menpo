from __future__ import division

import itertools
import warnings

import numpy as np

scipy_gaussian_filter = None  # expensive

from .base import ndfeature, imgfeature


@ndfeature
def gradient(pixels):
    r"""
    Calculates the gradient of an input image. The image is assumed to have
    channel information on the first axis. In the case of multiple channels,
    it returns the gradient over each axis over each channel as the first axis.

    The gradient is computed using second order accurate central differences in
    the interior and first order accurate one-side (forward or backwards)
    differences at the boundaries.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array where the first dimension
        is interpreted as channels. This means an N-dimensional image is
        represented by an N+1 dimensional array.
        If the image is 2-dimensional the pixels should be of type
        float/double (int is not supported).

    Returns
    -------
    gradient : `ndarray`
        The gradient over each axis over each channel. Therefore, the
        first axis of the gradient of a 2D, single channel image, will have
        length `2`. The first axis of the gradient of a 2D, 3-channel image,
        will have length `6`, the ordering being
        ``I[:, 0, 0] = [R0_y, G0_y, B0_y, R0_x, G0_x, B0_x]``. To be clear,
        all the ``y``-gradients are returned over each channel, then all
        the ``x``-gradients.
    """
    if pixels.dtype == np.uint8:
        raise TypeError("Attempting to take the gradient on a uint8 image.")
    n_dims = pixels.ndim - 1
    grad_per_dim_per_channel = [np.gradient(g, edge_order=1)
                                for g in pixels]
    # Flatten out the separate dims
    grad_per_channel = list(itertools.chain.from_iterable(
        grad_per_dim_per_channel))
    # Add a channel axis for broadcasting
    grad_per_channel = [g[None, ...] for g in grad_per_channel]

    # Permute the list so it is first axis, second axis, etc
    grad_per_channel = [grad_per_channel[i::n_dims]
                        for i in range(n_dims)]
    grad_per_channel = list(itertools.chain.from_iterable(grad_per_channel))

    # Concatenate gradient list into an array (the new_image)
    return np.concatenate(grad_per_channel, axis=0)


@ndfeature
def gaussian_filter(pixels, sigma):
    r"""
    Calculates the convolution of the input image with a multidimensional
    Gaussian filter.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    sigma : `float` or `list` of `float`
        The standard deviation for Gaussian kernel. The standard deviations of
        the Gaussian filter are given for each axis as a `list`, or as a single
        `float`, in which case it is equal for all axes.

    Returns
    -------
    output_image : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The filtered image has the same type and size as the input ``pixels``.
    """
    global scipy_gaussian_filter
    if scipy_gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
    output = np.empty(pixels.shape, dtype=pixels.dtype)
    for dim in range(pixels.shape[0]):
        scipy_gaussian_filter(pixels[dim], sigma, output=output[dim])
    return output


@ndfeature
def igo(pixels, double_angles=False, verbose=False):
    r"""
    Extracts Image Gradient Orientation (IGO) features from the input image.
    The output image has ``N * C`` number of channels, where ``N`` is the
    number of channels of the original image and ``C = 2`` or ``C = 4``
    depending on whether double angles are used.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    double_angles : `bool`, optional
        Assume that ``phi`` represents the gradient orientations.

        If this flag is ``False``, the features image is the concatenation of
        ``cos(phi)`` and ``sin(phi)``, thus 2 channels.

        If ``True``, the features image is the concatenation of
        ``cos(phi)``, ``sin(phi)``, ``cos(2 * phi)``, ``sin(2 * phi)``, thus 4
        channels.
    verbose : `bool`, optional
        Flag to print IGO related information.

    Returns
    -------
    igo : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The IGO features image. It has the same type and shape as the input
        ``pixels``. The output number of channels depends on the
        ``double_angles`` flag.

    Raises
    ------
    ValueError
        Image has to be 2D in order to extract IGOs.

    References
    ----------
    .. [1] G. Tzimiropoulos, S. Zafeiriou and M. Pantic, "Subspace learning
        from image gradient orientations", IEEE Transactions on Pattern Analysis
        and Machine Intelligence, vol. 34, num. 12, p. 2454--2466, 2012.
    """
    # check number of dimensions
    if len(pixels.shape) != 3:
        raise ValueError('IGOs only work on 2D images. Expects image data '
                         'to be 3D, channels + shape.')
    n_img_chnls = pixels.shape[0]
    # feature channels per image channel
    feat_chnls = 2
    if double_angles:
        feat_chnls = 4

    # compute gradients
    grad = gradient(pixels)
    # compute angles
    grad_orient = np.angle(grad[:n_img_chnls] + 1j * grad[n_img_chnls:])
    # compute igo image
    igo_pixels = np.empty((n_img_chnls * feat_chnls,
                           pixels.shape[1], pixels.shape[2]),
                          dtype=pixels.dtype)

    if double_angles:
        dbl_grad_orient = 2 * grad_orient
        # y angles
        igo_pixels[:n_img_chnls] = np.sin(grad_orient)
        igo_pixels[n_img_chnls:n_img_chnls * 2] = np.sin(dbl_grad_orient)

        # x angles
        igo_pixels[n_img_chnls * 2:n_img_chnls * 3] = np.cos(grad_orient)
        igo_pixels[n_img_chnls * 3:] = np.cos(dbl_grad_orient)
    else:
        igo_pixels[:n_img_chnls] = np.sin(grad_orient)  # y
        igo_pixels[n_img_chnls:] = np.cos(grad_orient)  # x

    # print information
    if verbose:
        info_str = "IGO Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[2], pixels.shape[1], n_img_chnls)
        info_str = "{}  - Double angles are {}.\n".format(
            info_str, 'enabled' if double_angles else 'disabled')
        info_str = "{}Output image size {}W x {}H with {} channels.".format(
            info_str, igo_pixels.shape[2], igo_pixels.shape[1], n_img_chnls)
        print(info_str)
    return igo_pixels


@ndfeature
def es(pixels, verbose=False):
    r"""
    Extracts Edge Structure (ES) features from the input image. The output image
    has ``N * C`` number of channels, where ``N`` is the number of channels of
    the original image and ``C = 2``.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either an image object itself or an array where the first axis
        represents the number of channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    verbose : `bool`, optional
        Flag to print ES related information.

    Returns
    -------
    es : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The ES features image. It has the same type and shape as the input
        ``pixels``. The output number of channels is ``C = 2``.

    Raises
    ------
    ValueError
        Image has to be 2D in order to extract ES features.

    References
    ----------
    .. [1] T. Cootes, C. Taylor, "On representing edge structure for model
        matching", Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (CVPR), 2001.
    """
    # check number of dimensions
    if len(pixels.shape) != 3:
        raise ValueError('ES features only work on 2D images. Expects '
                         'image data to be 3D, channels + shape.')
    n_img_chnls = pixels.shape[0]
    # feature channels per image channel
    feat_channels = 2
    # compute gradients
    grad = gradient(pixels)
    # compute magnitude
    grad_abs = np.abs(grad[:n_img_chnls] + 1j * grad[n_img_chnls:])
    # compute es image
    grad_abs = grad_abs + np.median(grad_abs)
    es_pixels = np.empty((pixels.shape[0] * feat_channels,
                          pixels.shape[1], pixels.shape[2]),
                         dtype=pixels.dtype)

    es_pixels[:n_img_chnls] = grad[:n_img_chnls] / grad_abs
    es_pixels[n_img_chnls:] = grad[n_img_chnls:] / grad_abs

    # print information
    if verbose:
        info_str = "ES Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[2], pixels.shape[1], n_img_chnls)
        info_str = "{}Output image size {}W x {}H with {} channels.".format(
            info_str, es_pixels.shape[2], es_pixels.shape[1], n_img_chnls)
        print(info_str)
    return es_pixels


@ndfeature
def daisy(pixels, step=1, radius=15, rings=2, histograms=2, orientations=8,
          normalization='l1', sigmas=None, ring_radii=None, verbose=False):
    r"""
    Extracts Daisy features from the input image. The output image has ``N * C``
    number of channels, where ``N`` is the number of channels of the original
    image and ``C`` is the feature channels determined by the input options.
    Specifically, ``C = (rings * histograms + 1) * orientations``.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    step : `int`, optional
        The sampling step that defines the density of the output image.
    radius : `int`, optional
        The radius (in pixels) of the outermost ring.
    rings : `int`, optional
        The number of rings to be used.
    histograms : `int`, optional
        The number of histograms sampled per ring.
    orientations : `int`, optional
        The number of orientations (bins) per histogram.
    normalization : [ 'l1', 'l2', 'daisy', None ], optional
        It defines how to normalize the descriptors
        If 'l1' then L1-normalization is applied at each descriptor.
        If 'l2' then L2-normalization is applied at each descriptor.
        If 'daisy' then L2-normalization is applied at individual histograms.
        If None then no normalization is employed.
    sigmas : `list` of `float` or ``None``, optional
        Standard deviation of spatial Gaussian smoothing for the centre
        histogram and for each ring of histograms. The `list` of sigmas should
        be sorted from the centre and out. I.e. the first sigma value defines
        the spatial smoothing of the centre histogram and the last sigma value
        defines the spatial smoothing of the outermost ring. Specifying sigmas
        overrides the `rings` parameter by setting ``rings = len(sigmas) - 1``.
    ring_radii : `list` of `float` or ``None``, optional
        Radius (in pixels) for each ring. Specifying `ring_radii` overrides the
        `rings` and `radius` parameters by setting ``rings = len(ring_radii)``
        and ``radius = ring_radii[-1]``.

        If both sigmas and ring_radii are given, they must satisfy ::

            len(ring_radii) == len(sigmas) + 1

        since no radius is needed for the centre histogram.
    verbose : `bool`
        Flag to print Daisy related information.

    Returns
    -------
    daisy : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The ES features image. It has the same type and shape as the input
        ``pixels``. The output number of channels is
        ``C = (rings * histograms + 1) * orientations``.

    Raises
    ------
    ValueError
        len(sigmas)-1 != len(ring_radii)
    ValueError
        Invalid normalization method.

    References
    ----------
    .. [1] E. Tola, V. Lepetit and P. Fua, "Daisy: An efficient dense descriptor
        applied to wide-baseline stereo", IEEE Transactions on Pattern Analysis
        and Machine Intelligence, vol. 32, num. 5, p. 815-830, 2010.
    """
    from menpo.external.skimage._daisy import _daisy

    # Parse options
    if sigmas is not None and ring_radii is not None \
            and len(sigmas) - 1 != len(ring_radii):
        raise ValueError('`len(sigmas)-1 != len(ring_radii)`')
    if ring_radii is not None:
        rings = len(ring_radii)
        radius = ring_radii[-1]
    if sigmas is not None:
        rings = len(sigmas) - 1
    if sigmas is None:
        sigmas = [radius * (i + 1) / float(2 * rings) for i in range(rings)]
    if ring_radii is None:
        ring_radii = [radius * (i + 1) / float(rings) for i in range(rings)]
    if normalization is None:
        normalization = 'off'
    if normalization not in ['l1', 'l2', 'daisy', 'off']:
        raise ValueError('Invalid normalization method.')

    # Compute daisy features
    daisy_descriptor = _daisy(pixels, step=step, radius=radius, rings=rings,
                              histograms=histograms, orientations=orientations,
                              normalization=normalization, sigmas=sigmas,
                              ring_radii=ring_radii)

    # print information
    if verbose:
        info_str = "Daisy Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[2], pixels.shape[1], pixels.shape[0])
        info_str = "{}  - Sampling step is {}.\n".format(info_str, step)
        info_str = "{}  - Radius of {} pixels, {} rings and {} histograms " \
                   "with {} orientations.\n".format(
            info_str, radius, rings, histograms, orientations)
        if not normalization == 'off':
            info_str = "{}  - Using {} normalization.\n".format(info_str,
                                                                normalization)
        else:
            info_str = "{}  - No normalization emplyed.\n".format(info_str)
        info_str = "{}Output image size {}W x {}H x {}.".format(
            info_str, daisy_descriptor.shape[2], daisy_descriptor.shape[1],
            daisy_descriptor.shape[0])
        print(info_str)

    return daisy_descriptor


@imgfeature
def normalize(img, scale_func=None, mode='all',
              error_on_divide_by_zero=True):
    r"""
    Normalize the pixel values via mean centering and an optional scaling. By
    default the scaling will be ``1.0``. The ``mode`` parameter selects
    whether the normalisation is computed across all pixels in the image or
    per-channel.

    Parameters
    ----------
    img : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    scale_func : `callable`, optional
        Compute the scaling factor. Expects a single parameter and an optional
        `axis` keyword argument and will be passed the entire pixel array.
        Should return a 1D numpy array of one or more values.
    mode : ``{all, per_channel}``, optional
        If ``all``, the normalization is over all channels. If
        ``per_channel``, each channel individually is mean centred and
        normalized in variance.
    error_on_divide_by_zero : `bool`, optional
        If ``True``, will raise a ``ValueError`` on dividing by zero.
        If ``False``, will merely raise a warning and only those values
        with non-zero denominators will be normalized.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A normalized copy of the image that was passed in.

    Raises
    ------
    ValueError
        If any of the denominators are 0 and ``error_on_divide_by_zero`` is
        ``True``.
    """
    if scale_func is None:
        def scale_func(_, axis=None):
            return np.array([1.0])

    pixels = img.as_vector(keep_channels=True)

    if mode == 'all':
        centered_pixels = pixels - np.mean(pixels)
        scale_factor = scale_func(centered_pixels)
    elif mode == 'per_channel':
        centered_pixels = pixels - np.mean(pixels, axis=1, keepdims=True)
        scale_factor = scale_func(centered_pixels, axis=1).reshape([-1, 1])
    else:
        raise ValueError("Supported modes are {{'all', 'per_channel'}} - '{}' "
                         "is not known".format(mode))

    zero_denom = (scale_factor == 0).ravel()
    any_non_zero = np.any(zero_denom)
    if error_on_divide_by_zero and any_non_zero:
        raise ValueError("Computed scale factor cannot be 0.0")
    elif any_non_zero:
        warnings.warn('One or more the scale factors are 0.0 and thus these'
                      'entries will be skipped during normalization.')
        non_zero_denom = ~zero_denom
        centered_pixels[non_zero_denom] = (centered_pixels[non_zero_denom] /
                                           scale_factor[non_zero_denom])
        return img.from_vector(centered_pixels)
    else:
        return img.from_vector(centered_pixels / scale_factor)


@ndfeature
def normalize_norm(pixels, mode='all', error_on_divide_by_zero=True):
    r"""
    Normalize the pixels to be mean centred and have unit norm. The ``mode``
    parameter selects whether the normalisation is computed across all pixels in
    the image or per-channel.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    mode : ``{all, per_channel}``, optional
        If ``all``, the normalization is over all channels. If
        ``per_channel``, each channel individually is mean centred and
        normalized in variance.
    error_on_divide_by_zero : `bool`, optional
        If ``True``, will raise a ``ValueError`` on dividing by zero.
        If ``False``, will merely raise a warning and only those values
        with non-zero denominators will be normalized.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A normalized copy of the image that was passed in.

    Raises
    ------
    ValueError
        If any of the denominators are 0 and ``error_on_divide_by_zero`` is
        ``True``.
    """

    def unit_norm(x, axis=None):
        return np.linalg.norm(x, axis=axis)

    return normalize(pixels, scale_func=unit_norm, mode=mode,
                     error_on_divide_by_zero=error_on_divide_by_zero)


@ndfeature
def normalize_std(pixels, mode='all', error_on_divide_by_zero=True):
    r"""
    Normalize the pixels to be mean centred and have unit standard deviation.
    The ``mode`` parameter selects whether the normalisation is computed across
    all pixels in the image or per-channel.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    mode : ``{all, per_channel}``, optional
        If ``all``, the normalization is over all channels. If
        ``per_channel``, each channel individually is mean centred and
        normalized in variance.
    error_on_divide_by_zero : `bool`, optional
        If ``True``, will raise a ``ValueError`` on dividing by zero.
        If ``False``, will merely raise a warning and only those values
        with non-zero denominators will be normalized.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A normalized copy of the image that was passed in.

    Raises
    ------
    ValueError
        If any of the denominators are 0 and ``error_on_divide_by_zero`` is
        ``True``.
    """

    def unit_std(x, axis=None):
        return np.std(x, axis=axis)

    return normalize(pixels, scale_func=unit_std, mode=mode,
                     error_on_divide_by_zero=error_on_divide_by_zero)


@ndfeature
def normalize_var(pixels, mode='all', error_on_divide_by_zero=True):
    r"""
    Normalize the pixels to be mean centred and normalize according
    to the variance.
    The ``mode`` parameter selects whether the normalisation is computed across
    all pixels in the image or per-channel.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    mode : ``{all, per_channel}``, optional
        If ``all``, the normalization is over all channels. If
        ``per_channel``, each channel individually is mean centred and
        normalized in variance.
    error_on_divide_by_zero : `bool`, optional
        If ``True``, will raise a ``ValueError`` on dividing by zero.
        If ``False``, will merely raise a warning and only those values
        with non-zero denominators will be normalized.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A normalized copy of the image that was passed in.

    Raises
    ------
    ValueError
        If any of the denominators are 0 and ``error_on_divide_by_zero`` is
        ``True``.
    """

    def unit_var(x, axis=None):
        return np.var(x, axis=axis)

    return normalize(pixels, scale_func=unit_var, mode=mode,
                     error_on_divide_by_zero=error_on_divide_by_zero)


@ndfeature
def no_op(pixels):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A copy of the image that was passed in.
    """
    return pixels.copy()


def features_selection_widget():
    r"""
    Widget that allows for easy selection of a features function and its
    options. It also has a 'preview' tab for visual inspection. It returns a
    `list` of length 1 with the selected features function closure.

    Returns
    -------
    features_function : `list` of length ``1``
        The function closure of the features function using `functools.partial`.
        So the function can be called as: ::

            features_image = features_function[0](image)

    Examples
    --------
    The widget can be invoked as ::

        from menpo.feature import features_selection_widget
        features_fun = features_selection_widget()

    And the returned function can be used as ::

        import menpo.io as mio
        image = mio.import_builtin_asset.lenna_png()
        features_image = features_fun[0](image)
    """
    from menpowidgets import features_selection

    return features_selection()
