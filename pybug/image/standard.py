from copy import deepcopy
import numpy as np
import PIL.Image as PILImage
import scipy.linalg


def as_PILImage(image):
    r"""
    Return a PIL copy of the image. Scales the image by ``255`` and
    converts to ``np.uint8``.

    Returns
    -------
    pil_image : ``PILImage``
        PIL copy of image as ``np.uint8``
    """
    return PILImage.fromarray((image.pixels * 255).astype(np.uint8))


def as_greyscale(image, mode='luminosity', channel=None):
    r"""
    Returns a greyscale version of the RGB image.

    Parameters
    ----------
    mode : {'average', 'luminosity', 'channel'}
        'luminosity' - Calculates the luminance using the CCIR 601 formula
            ``Y' = 0.2989 R' + 0.5870 G' + 0.1140 B'``
        'average' - intensity is an equal average of all three channels
        'channel' - a specific channel is used

        Default 'luminosity'

    channel: int, optional
        The channel to be taken. Only used if mode is 'channel'.

        Default: None

    Returns
    -------
    greyscale_image: :class:`IntensityImage`
        A copy of this image in greyscale.
    """
    pixels = None
    if mode == 'luminosity':
        # Invert the transformation matrix to get more precise values
        T = scipy.linalg.inv(np.array([[1.0, 0.956, 0.621],
                                       [1.0, -0.272, -0.647],
                                       [1.0, -1.106, 1.703]]))
        coef = T[0, :]
        pixels = np.dot(image.pixels, coef.T)
    elif mode == 'average':
        pixels = np.mean(image.pixels, axis=-1)
    elif mode == 'channel':
        if channel is None:
            raise ValueError("for the 'channel' mode you have to provide"
                             " a channel index")
        elif channel < 0 or channel > 2:
            raise ValueError("channel can only be 0, 1, or 2 "
                             "in RGB images.")
        pixels = image.pixels[..., channel]
    mask = deepcopy(image.mask)
    greyscale = IntensityImage(pixels, mask=mask)
    greyscale.landmarks = image.landmarks
    return greyscale
