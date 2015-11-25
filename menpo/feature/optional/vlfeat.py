from __future__ import division
from functools import partial
import numpy as np
from menpo.base import MenpoMissingDependencyError
from ..base import winitfeature

try:
    from cyvlfeat.sift.dsift import dsift as cyvlfeat_dsift
except ImportError:
    raise MenpoMissingDependencyError('cyvlfeat')


@winitfeature
def dsift(pixels, window_step_horizontal=1, window_step_vertical=1,
          num_bins_horizontal=2, num_bins_vertical=2, num_or_bins=9,
          cell_size_horizontal=6, cell_size_vertical=6, fast=True,
          verbose=False):
    r"""
    Computes a 2-dimensional dense SIFT features image with ``C`` number of
    channels, where
    ``C = num_bins_horizontal * num_bins_vertical * num_or_bins``. The dense
    SIFT [2]_ implementation is taken from Vlfeat [1]_.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, Y, X)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels.
    window_step_horizontal : `int`, optional
        Defines the horizontal step by which the window is moved, thus it
        controls the features density. The metric unit is pixels.
    window_step_vertical : `int`, optional
        Defines the vertical step by which the window is moved, thus it
        controls the features density. The metric unit is pixels.
    num_bins_horizontal : `int`, optional
        Defines the number of histogram bins in the X direction.
    num_bins_vertical : `int`, optional
        Defines the number of histogram bins in the Y direction.
    num_or_bins : `int`, optional
        Defines the number of orientation histogram bins.
    cell_size_horizontal : `int`, optional
        Defines cell width in pixels. The cell is the region that is covered by
        a spatial bin.
    cell_size_vertical : `int`, optional
        Defines cell height in pixels. The cell is the region that is covered by
        a spatial bin.
    fast : `bool`, optional
        If ``True``, then the windowing function is a piecewise-flat, rather
        than Gaussian. While this breaks exact SIFT equivalence, in practice it
        is much faster to compute.
    verbose : `bool`, optional
        Flag to print SIFT related information.

    Raises
    ------
    ValueError
        Only 2D arrays are supported
    ValueError
        Size must only contain positive integers.
    ValueError
        Step must only contain positive integers.
    ValueError
        Window size must be a positive integer.
    ValueError
        Geometry must only contain positive integers.

    References
    ----------
    .. [1] Vedaldi, Andrea, and Brian Fulkerson. "VLFeat: An open and portable
       library of computer vision algorithms." Proceedings of the international
       conference on Multimedia. ACM, 2010.
    .. [2] Lowe, David G. "Distinctive image features from scale-invariant
       keypoints." International journal of computer vision 60.2 (2004): 91-110.
    """
    # If norm is set to True, then the centers array will have a third column
    # with descriptor norm, or energy, before contrast normalization.
    # This information can be used to suppress low contrast descriptors.
    centers, output = cyvlfeat_dsift(
        pixels[0], step=[window_step_vertical, window_step_horizontal],
        size=[cell_size_vertical, cell_size_horizontal], bounds=None,
        norm=False, fast=fast, float_descriptors=True,
        geometry=(num_bins_vertical, num_bins_horizontal, num_or_bins),
        verbose=False)

    # the output shape can be calculated from looking at the range of
    # centres / the window step size in each dimension. Note that cyvlfeat
    # returns x, y centres.
    shape = (((centers[-1, :] - centers[0, :]) /
              [window_step_vertical, window_step_horizontal]) + 1)

    # print information
    if verbose:
        info_str = "Dense SIFT features:\n" \
                   "  - Input image is {}W x {}H with {} channels.\n" \
                   "  - Sampling step of ({}W, {}H).\n" \
                   "  - {}W x {}H spatial bins and {} orientation bins.\n" \
                   "  - Cell size of {}W x {}H pixels.\n".format(
                   pixels.shape[2], pixels.shape[1], pixels.shape[0],
                   window_step_horizontal, window_step_vertical,
                   num_bins_horizontal, num_bins_vertical, num_or_bins,
                   cell_size_horizontal, cell_size_vertical)
        if fast:
            info_str += "  - Fast mode is enabled.\n"
        info_str += "Output image size {}W x {}H x {}.".format(
            int(shape[1]), int(shape[0]), output.shape[0])
        print(info_str)

    # return SIFT and centers in the correct form
    return (np.require(np.rollaxis(output.reshape((shape[0], shape[1], -1)),
                                   -1),
                       dtype=np.double, requirements=['C']),
            np.require(centers.reshape((shape[0], shape[1], -1)),
                       dtype=np.int))


# A predefined method for a 'faster' dsift method
fast_dsift = partial(dsift, fast=True, cell_size_vertical=5,
                     cell_size_horizontal=5, num_bins_horizontal=1,
                     num_bins_vertical=1, num_or_bins=8)
fast_dsift.__name__ = 'fast_dsift'
fast_dsift.__doc__ = dsift.__doc__


# Predefined dsift that returns a 128d vector
def vector_128_dsift(x, dtype=np.float32):
    r"""
    Computes a SIFT feature vector from a square patch (or image). Patch
    **must** be square and the output vector will *always* be a ``(128,)``
    vector. Please see :func:`dsift` for more information.

    Parameters
    ----------
    x : :map:`Image` or subclass or ``(C, Y, Y)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. Must be square i.e.
        ``height == width``.
    dtype : ``np.dtype``, optional
        The dtype of the returned vector.

    Raises
    ------
    ValueError
        Only square images are supported.
    """
    if not isinstance(x, np.ndarray):
        x = x.pixels
    if x.shape[-1] != x.shape[-2]:
        raise ValueError('This feature only works with square images '
                         'i.e. width == height')
    patch_shape = x.shape[-1]
    n_bins = 4
    c_size = patch_shape // n_bins
    if x.dtype == np.uint8:
        x *= (1.0 / 255.0)
    return dsift(x,
                 window_step_horizontal=patch_shape,
                 window_step_vertical=patch_shape,
                 num_bins_horizontal=n_bins, num_bins_vertical=n_bins,
                 cell_size_horizontal=c_size, cell_size_vertical=c_size,
                 num_or_bins=8, fast=True).astype(dtype)


# Predefined dsift that returns a 128d vector normalized by the hellinger norm
def hellinger_vector_128_dsift(x):
    r"""
    Computes a SIFT feature vector from a square patch (or image). Patch
    **must** be square and the output vector will *always* be a ``(128,)``
    vector. Please see :func:`dsift` for more information.

    The output of :func:`vector_128_dsift` is normalised using the hellinger
    norm (also called the Bhattacharyya distance) which is a measure
    designed to quantify the similarity between two probability distributions.
    Since SIFT is a histogram based feature, this has been shown to improve
    performance. Please see [1]_ for more information.

    Parameters
    ----------
    x : :map:`Image` or subclass or ``(C, Y, Y)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels. Must be square i.e.
        ``height == width``.
    dtype : ``np.dtype``, optional
        The dtype of the returned vector.

    Raises
    ------
    ValueError
        Only square images are supported.

    References
    ----------
    .. [1] Arandjelovic, Relja, and Andrew Zisserman. "Three things everyone
           should know to improve object retrieval.", CVPR, 2012.
    """
    h = vector_128_dsift(x)
    h /= (h.sum(axis=0) + 1e-15)
    return np.sqrt(h)
