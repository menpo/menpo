import itertools
import numpy as np
scipy_gaussian_filter = None  # expensive

from .base import ndfeature, winitfeature
from .windowiterator import WindowIterator


@ndfeature
def gradient(pixels):
    r"""
    Calculates the gradient of an input image. The image is assumed to have
    channel information on the last axis. In the case of multiple channels,
    it returns the gradient over each axis over each channel as the last axis.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.

    Returns
    -------
    gradient : ndarray, shape (X, Y, ..., Z, C * length([X, Y, ..., Z]))
        The gradient over each axis over each channel. Therefore, the
        last axis of the gradient of a 2D, single channel image, will have
        length `2`. The last axis of the gradient of a 2D, 3-channel image,
        will have length `6`, he ordering being [Rd_x, Rd_y, Gd_x, Gd_y,
        Bd_x, Bd_y].

    """
    grad_per_dim_per_channel = [np.gradient(g, edge_order=1) for g in
                                np.rollaxis(pixels, -1)]
    # Flatten out the separate dims
    grad_per_channel = list(itertools.chain.from_iterable(
        grad_per_dim_per_channel))
    # Add a channel axis for broadcasting
    grad_per_channel = [g[..., None] for g in grad_per_channel]
    # Concatenate gradient list into an array (the new_image)
    return np.concatenate(grad_per_channel, axis=-1)


@ndfeature
def gaussian_filter(pixels, sigma):
    r"""
    Calculates the convolution of the input image with a multidimensional
    Gaussian filter.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
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
    output = np.empty(pixels.shape)
    for dim in range(pixels.shape[2]):
        scipy_gaussian_filter(pixels[..., dim], sigma, output=output[..., dim])
    return output


@winitfeature
def hog(pixels, mode='dense', algorithm='dalaltriggs', num_bins=9,
        cell_size=8, block_size=2, signed_gradient=True, l2_norm_clip=0.2,
        window_height=1, window_width=1, window_unit='blocks',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False):
    r"""
    Extracts Histograms of Oriented Gradients (HOG) features from the input
    image.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    mode : {``dense``, ``sparse``}, optional
        The ``sparse`` case refers to the traditional usage of HOGs, so
        predefined parameters values are used.

        The ``sparse`` case of ``dalaltriggs`` algorithm sets
        ``window_height = window_width = block_size`` and
        ``window_step_horizontal = window_step_vertical = cell_size``.

        The ``sparse`` case of ``zhuramanan`` algorithm sets
        ``window_height = window_width = 3 * cell_size`` and
        ``window_step_horizontal = window_step_vertical = cell_size``.

        In the ``dense`` case, the user can choose values for `window_height`,
        `window_width`, `window_unit`, `window_step_vertical`,
        `window_step_horizontal`, `window_step_unit` and `padding` to customize
        the HOG calculation.
    window_height : `float`, optional
        Defines the height of the window. The metric unit is defined by
        `window_unit`.
    window_width : `float`, optional
        Defines the width of the window. The metric unit is defined by
        `window_unit`.
    window_unit : {``blocks``, ``pixels``}, optional
        Defines the metric unit of the `window_height` and `window_width`
        parameters.
    window_step_vertical : `float`, optional
        Defines the vertical step by which the window is moved, thus it
        controls the features' density. The metric unit is defined by
        `window_step_unit`.
    window_step_horizontal : `float`, optional
        Defines the horizontal step by which the window is moved, thus it
        controls the features' density. The metric unit is defined by
        `window_step_unit`.
    window_step_unit : {``pixels``, ``cells``}, optional
        Defines the metric unit of the `window_step_vertical` and
        `window_step_horizontal` parameters.
    padding : `bool`, optional
        If ``True``, the output image is padded with zeros to match the input
        image's size.
    algorithm : {``dalaltriggs``, ``zhuramanan``}, optional
        Specifies the algorithm used to compute HOGs. ``dalaltriggs`` is the
        implementation of [1] and ``zhuramanan`` is the implementation of [2].
    cell_size : `float`, optional
        Defines the cell size in pixels. This value is set to both the width
        and height of the cell. This option is valid for both algorithms.
    block_size : `float`, optional
        Defines the block size in cells. This value is set to both the width
        and height of the block. This option is valid only for the
        ``dalaltriggs`` algorithm.
    num_bins : `float`, optional
        Defines the number of orientation histogram bins. This option is
        valid only for the ``dalaltriggs`` algorithm.
    signed_gradient : `bool`, optional
        Flag that defines whether we use signed or unsigned gradient angles.
        This option is valid only for the ``dalaltriggs`` algorithm.
    l2_norm_clip : `float`, optional
        Defines the clipping value of the gradients' L2-norm. This option is
        valid only for the ``dalaltriggs`` algorithm.
    verbose : `bool`, optional
        Flag to print HOG related information.

    Returns
    -------
    hog : :map:`Image` or subclass or ``(X, Y, ..., Z, K)`` `ndarray`
        The HOG features image. It has the same type as the input ``pixels``.
        The output number of channels in the case of ``dalaltriggs`` is
        ``K = num_bins * block_size *block_size`` and ``K = 31`` in the case of
        ``zhuramanan``.

    Raises
    ------
    ValueError
        HOG features mode must be either dense or sparse
    ValueError
        Algorithm must be either dalaltriggs or zhuramanan
    ValueError
        Number of orientation bins must be > 0
    ValueError
        Cell size (in pixels) must be > 0
    ValueError
        Block size (in cells) must be > 0
    ValueError
        Value for L2-norm clipping must be > 0.0
    ValueError
        Window height must be >= block size and <= image height
    ValueError
        Window width must be >= block size and <= image width
    ValueError
        Window unit must be either pixels or blocks
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or cells

    References
    ----------
    .. [1] N. Dalal and B. Triggs, "Histograms of oriented gradients for human
        detection", Proceedings of the IEEE Conference on Computer Vision and
        Pattern Recognition (CVPR), 2005.
    .. [2] X. Zhu, D. Ramanan. "Face detection, pose estimation and landmark
        localization in the wild", Proceedings of the IEEE Conference on
        Computer Vision and Pattern Recognition (CVPR), 2012.
    """
    # Parse options
    if mode not in ['dense', 'sparse']:
        raise ValueError("HOG features mode must be either dense or sparse")
    if algorithm not in ['dalaltriggs', 'zhuramanan']:
        raise ValueError("Algorithm must be either dalaltriggs or zhuramanan")
    if num_bins <= 0:
        raise ValueError("Number of orientation bins must be > 0")
    if cell_size <= 0:
        raise ValueError("Cell size (in pixels) must be > 0")
    if block_size <= 0:
        raise ValueError("Block size (in cells) must be > 0")
    if l2_norm_clip <= 0.0:
        raise ValueError("Value for L2-norm clipping must be > 0.0")
    if mode == 'dense':
        if window_unit not in ['pixels', 'blocks']:
            raise ValueError("Window unit must be either pixels or blocks")
        window_height_temp = window_height
        window_width_temp = window_width
        if window_unit == 'blocks':
            window_height_temp = window_height * block_size * cell_size
            window_width_temp = window_width * block_size * cell_size
        if (window_height_temp < block_size * cell_size or
            window_height_temp > pixels.shape[0]):
            raise ValueError("Window height must be >= block size and <= "
                             "image height")
        if (window_width_temp < block_size*cell_size or
            window_width_temp > pixels.shape[1]):
            raise ValueError("Window width must be >= block size and <= "
                             "image width")
        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")
        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")
        if window_step_unit not in ['pixels', 'cells']:
            raise ValueError("Window step unit must be either pixels or cells")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)
    pixels *= 255.

    # Dense case
    if mode == 'dense':
        # Iterator parameters
        if algorithm == 'dalaltriggs':
            algorithm = 1
            if window_unit == 'blocks':
                block_in_pixels = cell_size * block_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        elif algorithm == 'zhuramanan':
            algorithm = 2
            if window_unit == 'blocks':
                block_in_pixels = 3 * cell_size
                window_height = np.uint32(window_height * block_in_pixels)
                window_width = np.uint32(window_width * block_in_pixels)
            if window_step_unit == 'cells':
                window_step_vertical = np.uint32(window_step_vertical *
                                                 cell_size)
                window_step_horizontal = np.uint32(window_step_horizontal *
                                                   cell_size)
        iterator = WindowIterator(pixels, window_height, window_width,
                                  window_step_horizontal,
                                  window_step_vertical, padding)
    # Sparse case
    else:
        # Create iterator
        if algorithm == 'dalaltriggs':
            algorithm = 1
            window_size = cell_size * block_size
            step = cell_size
        else:
            algorithm = 2
            window_size = 3 * cell_size
            step = cell_size
        iterator = WindowIterator(pixels, window_size, window_size, step,
                                  step, False)
    # Print iterator's info
    if verbose:
        print(iterator)
    # Compute HOG
    return iterator.HOG(algorithm, num_bins, cell_size, block_size,
                        signed_gradient, l2_norm_clip, verbose)


@ndfeature
def igo(pixels, double_angles=False, verbose=False):
    r"""
    Extracts Image Gradient Orientation (IGO) features from the input image.
    The output image has ``N * C`` number of channels, where ``N`` is the
    number of channels of the original image and ``C = 2`` or ``C = 4``
    depending on whether double angles are used.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
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
                         'to be 3D, shape + channels.')
    # feature channels per image channel
    feat_channels = 2
    if double_angles:
        feat_channels = 4
    # compute gradients
    grad = gradient(pixels)
    # compute angles
    grad_orient = np.angle(grad[..., ::2] + 1j * grad[..., 1::2])
    # compute igo image
    igo_pixels = np.empty((pixels.shape[0], pixels.shape[1],
                           pixels.shape[-1] * feat_channels))
    igo_pixels[..., ::feat_channels] = np.cos(grad_orient)
    igo_pixels[..., 1::feat_channels] = np.sin(grad_orient)
    if double_angles:
        igo_pixels[..., 2::feat_channels] = np.cos(2 * grad_orient)
        igo_pixels[..., 3::feat_channels] = np.sin(2 * grad_orient)

    # print information
    if verbose:
        info_str = "IGO Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[1], pixels.shape[0],
            pixels.shape[2])
        if double_angles:
            info_str = "{}  - Double angles are enabled.\n".format(info_str)
        else:
            info_str = "{}  - Double angles are disabled.\n".format(info_str)
        info_str = "{}Output image size {}W x {}H x {}.".format(
            info_str, igo_pixels.shape[1], igo_pixels.shape[0],
            igo_pixels.shape[2])
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
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
        dimension is interpreted as channels. This means an N-dimensional image
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
                         'image data to be 3D, shape + channels.')
    # feature channels per image channel
    feat_channels = 2
    # compute gradients
    grad = gradient(pixels)
    # compute magnitude
    grad_abs = np.abs(grad[..., ::2] + 1j * grad[..., 1::2])
    # compute es image
    grad_abs = grad_abs + np.median(grad_abs)
    es_pixels = np.empty((pixels.shape[0], pixels.shape[1],
                          pixels.shape[-1] * feat_channels))
    es_pixels[..., ::feat_channels] = grad[..., ::2] / grad_abs
    es_pixels[..., 1::feat_channels] = grad[..., 1::2] / grad_abs
    # print information
    if verbose:
        info_str = "ES Features:\n"
        info_str = "{}  - Input image is {}W x {}H with {} channels.\n".format(
            info_str, pixels.shape[1], pixels.shape[0],
            pixels.shape[2])
        info_str = "{}Output image size {}W x {}H x {}.".format(
            info_str, es_pixels.shape[1], es_pixels.shape[0],
            es_pixels.shape[2])
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
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
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
            info_str, pixels.shape[1], pixels.shape[0], pixels.shape[2])
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
            info_str, daisy_descriptor.shape[1], daisy_descriptor.shape[0],
            daisy_descriptor.shape[2])
        print(info_str)

    return daisy_descriptor


@winitfeature
def lbp(pixels, radius=None, samples=None, mapping_type='riu2',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding=True, verbose=False,
        skip_checks=False):
    r"""
    Extracts Local Binary Pattern (LBP) features from the input image. The
    output image has ``N * C`` number of channels, where ``N`` is the number of
    channels of the original image and ``C`` is the number of radius/samples
    values combinations that are used in the LBP computation.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.
    radius : `int` or `list` of `int` or ``None``, optional
        It defines the radius of the circle (or circles) at which the sampling
        points will be extracted. The radius (or radii) values must be greater
        than zero. There must be a radius value for each samples value, thus
        they both need to have the same length. If ``None``, then
        ``[1, 2, 3, 4]`` is used.
    samples : `int` or `list` of `int` or ``None``, optional
        It defines the number of sampling points that will be extracted at each
        circle. The samples value (or values) must be greater than zero. There
        must be a samples value for each radius value, thus they both need to
        have the same length. If ``None``, then ``[8, 8, 8, 8]`` is used.
    mapping_type : {``u2``, ``ri``, ``riu2``, ``none``}, optional
        It defines the mapping type of the LBP codes. Select ``u2`` for
        uniform-2 mapping, ``ri`` for rotation-invariant mapping, ``riu2`` for
        uniform-2 and rotation-invariant mapping and ``none`` to use no mapping
        and only the decimal values instead.
    window_step_vertical : `float`, optional
        Defines the vertical step by which the window is moved, thus it controls
        the features density. The metric unit is defined by `window_step_unit`.
    window_step_horizontal : `float`, optional
        Defines the horizontal step by which the window is moved, thus it
        controls the features density. The metric unit is defined by
        `window_step_unit`.
    window_step_unit : {``pixels``, ``window``}, optional
        Defines the metric unit of the `window_step_vertical` and
        `window_step_horizontal` parameters.
    padding : `bool`, optional
        If ``True``, the output image is padded with zeros to match the input
        image's size.
    verbose : `bool`, optional
        Flag to print LBP related information.
    skip_checks : `bool`, optional
        If ``True``, do not perform any validation of the parameters.

    Returns
    -------
    lbp : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        The ES features image. It has the same type and shape as the input
        ``pixels``. The output number of channels is
        ``C = len(radius) * len(samples)``.

    Raises
    ------
    ValueError
        Radius and samples must both be either integers or lists
    ValueError
        Radius and samples must have the same length
    ValueError
        Radius must be > 0
    ValueError
        Radii must be > 0
    ValueError
        Samples must be > 0
    ValueError
        Mapping type must be u2, ri, riu2 or none
    ValueError
        Horizontal window step must be > 0
    ValueError
        Vertical window step must be > 0
    ValueError
        Window step unit must be either pixels or window

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, and T. Maenpaa, "Multiresolution gray-scale
        and rotation invariant texture classification with local binary
        patterns", IEEE Transactions on Pattern Analysis and Machine
        Intelligence, vol. 24, num. 7, p. 971-987, 2002.
    """
    if radius is None:
        radius = range(1, 5)
    if samples is None:
        samples = [8]*4

    if not skip_checks:
        # Check parameters
        if ((isinstance(radius, int) and isinstance(samples, list)) or
                (isinstance(radius, list) and isinstance(samples, int))):
            raise ValueError("Radius and samples must both be either integers "
                             "or lists")
        elif isinstance(radius, list) and isinstance(samples, list):
            if len(radius) != len(samples):
                raise ValueError("Radius and samples must have the same "
                                 "length")

        if isinstance(radius, int) and radius < 1:
            raise ValueError("Radius must be > 0")
        elif isinstance(radius, list) and sum(r < 1 for r in radius) > 0:
            raise ValueError("Radii must be > 0")

        if isinstance(samples, int) and samples < 1:
            raise ValueError("Samples must be > 0")
        elif isinstance(samples, list) and sum(s < 1 for s in samples) > 0:
            raise ValueError("Samples must be > 0")

        if mapping_type not in ['u2', 'ri', 'riu2', 'none']:
            raise ValueError("Mapping type must be u2, ri, riu2 or "
                             "none")

        if window_step_horizontal <= 0:
            raise ValueError("Horizontal window step must be > 0")

        if window_step_vertical <= 0:
            raise ValueError("Vertical window step must be > 0")

        if window_step_unit not in ['pixels', 'window']:
            raise ValueError("Window step unit must be either pixels or "
                             "window")

    # Correct input image_data
    pixels = np.asfortranarray(pixels)

    # Parse options
    radius = np.asfortranarray(radius)
    samples = np.asfortranarray(samples)
    window_height = np.uint32(2 * radius.max() + 1)
    window_width = window_height
    if window_step_unit == 'window':
        window_step_vertical = np.uint32(window_step_vertical * window_height)
        window_step_horizontal = np.uint32(window_step_horizontal *
                                           window_width)
    if mapping_type == 'u2':
        mapping_type = 1
    elif mapping_type == 'ri':
        mapping_type = 2
    elif mapping_type == 'riu2':
        mapping_type = 3
    else:
        mapping_type = 0

    # Create iterator object
    iterator = WindowIterator(pixels, window_height, window_width,
                              window_step_horizontal, window_step_vertical,
                              padding)

    # Print iterator's info
    if verbose:
        print(iterator)

    # Compute LBP
    return iterator.LBP(radius, samples, mapping_type, verbose)


@ndfeature
def no_op(pixels):
    r"""
    A no operation feature - does nothing but return a copy of the pixels
    passed in.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        Either the image object itself or an array with the pixels. The last
        dimension is interpreted as channels. This means an N-dimensional image
        is represented by an N+1 dimensional array.

    Returns
    -------
    pixels : :map:`Image` or subclass or ``(X, Y, ..., Z, C)`` `ndarray`
        A copy of the image that was passed in.
    """
    return pixels.copy()


def features_selection_widget(popup=True):
    r"""
    Widget that allows for easy selection of a features function and its
    options. It also has a 'preview' tab for visual inspection. It returns a
    `list` of length 1 with the selected features function closure.

    Parameters
    ----------
    popup : `bool`, optional
        If ``True``, the widget will appear as a popup window.

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
    from menpo.visualize.widgets import features_selection

    return features_selection(popup=popup)
