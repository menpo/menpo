import numpy as np
from scipy.misc import imrotate

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


# TODO: Needs fixing ...
@ndfeature
def glyph(pixels, vectors_block_size=10, use_negative=False, channels=None):
    r"""
    Create the glyph of a feature image that can be used for visualization. If
    `pixels` have negative values, the `use_negative` flag controls whether
    there will be created a glyph of both positive and negative values
    concatenated the one on top of the other.

    Parameters
    ----------
    pixels : :map:`Image` or subclass or ``(C, X, Y, ..., Z)`` `ndarray`
        Either the image object itself or an array with the pixels. The first
        dimension is interpreted as channels.
    vectors_block_size : `int`
        Defines the size of each block with vectors of the glyph image.
    use_negative : `bool`
        Defines whether to take into account possible negative values of
        feature_data.
    channels : `list` of `int` or ``None``
        The list of channels to be used. If ``None``, then all the channels are
        employed.
    """
    # first, choose the appropriate channels
    if channels is None:
        pixels = pixels[:4]
    elif channels != 'all':
        pixels = pixels[channels]
    else:
        pixels = pixels

    # TODO: This is a temporal fix
    # flip axis
    pixels = np.rollaxis(pixels, 0, len(pixels.shape))

    # compute the glyph
    negative_weights = -pixels
    scale = np.maximum(pixels.max(), negative_weights.max())
    pos = _create_feature_glyph(pixels, vectors_block_size)
    pos = pos * 255 / scale
    glyph_image = pos
    if use_negative and pixels.min() < 0:
        neg = _create_feature_glyph(negative_weights, vectors_block_size)
        neg = neg * 255 / scale
        glyph_image = np.concatenate((pos, neg))
    # return as c-contiguous
    return np.ascontiguousarray(glyph_image[None, ...])  # add a channel axis


def _create_feature_glyph(feature, vbs):
    r"""
    Create glyph of feature pixels.

    Parameters
    ----------
    feature : ``(N, D)`` `ndarray`
        The feature pixels to use.
    vbs : `int`
        Defines the size of each block with vectors of the glyph image.
    """
    # vbs = Vector block size
    num_bins = feature.shape[2]
    # construct a "glyph" for each orientation
    block_image_temp = np.zeros((vbs, vbs))
    # Create a vertical line of ones, to be the first vector
    block_image_temp[:, np.round(vbs / 2) - 1:np.round(vbs / 2) + 1] = 1
    block_im = np.zeros((block_image_temp.shape[0],
                         block_image_temp.shape[1],
                         num_bins))
    # First vector as calculated above
    block_im[:, :, 0] = block_image_temp
    # Number of bins rotations to create an 'asterisk' shape
    for i in range(1, num_bins):
        block_im[:, :, i] = imrotate(block_image_temp, -i * vbs)

    # make pictures of positive feature_data by adding up weighted glyphs
    feature[feature < 0] = 0
    glyph_im = np.sum(block_im[None, None, :, :, :] *
                      feature[:, :, None, None, :], axis=-1)
    glyph_im = np.bmat(glyph_im.tolist())
    return glyph_im
