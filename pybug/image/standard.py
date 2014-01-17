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
