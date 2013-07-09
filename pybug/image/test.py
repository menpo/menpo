import numpy as np
from pybug.image import Image


def test_image_creation():
    pixels = np.ones((120, 120, 3))
    Image(pixels)
