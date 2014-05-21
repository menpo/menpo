import numpy as np
from menpo.image import *


def update_im_from_vector(im):
    new_values = np.random.random(im.pixels.shape)
    im.from_vector_inplace(new_values.flatten())
    assert im.pixels.shape == new_values.shape
    return new_values


def test_1channel_update_from_vector():
    im = MaskedImage.blank((10, 10))
    update_im_from_vector(im)


def test_3channel_update_from_vector():
    im = MaskedImage.blank((10, 10), n_channels=3)
    update_im_from_vector(im)


def test_maskedimage_update_from_vector():
    im = MaskedImage.blank((10, 10), n_channels=10)
    update_im_from_vector(im)
