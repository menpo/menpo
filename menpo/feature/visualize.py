import numpy as np

from .base import ndfeature


@ndfeature
def sum_channels(pixels, channels=None):
    r"""
    Create the sum of the channels of an image that can be used for
    visualization.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels.
    channels : `list` of `int` or ``None``
        The list of channels to be used. If ``None``, then all the channels are
        employed.
    """
    # if channels is None, then all channels are used
    if channels is None:
        # Not indexing is twice as fast
        sum_image = np.sum(pixels, axis=0)
    else:
        sum_image = np.sum(pixels[channels], axis=0)
    return sum_image.reshape((1,) + sum_image.shape)  # add a channel axis
