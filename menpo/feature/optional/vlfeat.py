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
    ``C = num_bins_horizontal * num_bins_vertical * num_or_bins``.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
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
                   "  - Sampling step of ({}W,{}H).\n" \
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
