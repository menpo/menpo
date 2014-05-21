import numpy as np
from menpo.image import *


def test_blank_1_channel_image():
    mask = np.zeros((10, 10), dtype=np.bool)
    im = MaskedImage.blank((10, 10), mask=mask)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 1
    assert np.all(im.mask.pixels == 0.0)

    im = MaskedImage.blank((10, 10), fill=0.5)
    assert np.all(im.pixels == 0.5)


def test_blank_3_channel_image():
    mask = np.zeros((10, 10), dtype=np.bool)
    im = MaskedImage.blank((10, 10), mask=mask, n_channels=3)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 3
    assert np.all(im.mask.pixels == 0.0)

    im = MaskedImage.blank((10, 10), fill=0.5, n_channels=3)
    assert np.all(im.pixels == 0.5)


def test_blank_maskedimage():
    mask = np.zeros((10, 10), dtype=np.bool)
    im = MaskedImage.blank((10, 10), mask=mask, n_channels=10)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 10
    assert np.all(im.mask.pixels == 0.0)

    im = MaskedImage.blank((10, 10), fill=2.0, n_channels=10)
    assert np.all(im.pixels == 2.0)
