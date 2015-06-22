from menpo.image import Image, MaskedImage
import numpy as np


def test_rescale_pixels():
    img = Image.init_blank((10, 10), n_channels=1)
    img.pixels[:, 6:, 6:] = 2

    img_rescaled = img.rescale_pixels(0, 255)
    assert np.min(img_rescaled.pixels) == 0
    assert np.max(img_rescaled.pixels) == 255
    assert np.all(img_rescaled.pixels[:, 6:, 6:] == 255)


def test_rescale_pixels_all_channels():
    img = Image.init_blank((10, 10), n_channels=2)
    img.pixels[0, 6:, 6:] = 2
    img.pixels[1, 6:, 6:] = 4

    img_rescaled = img.rescale_pixels(0, 100, per_channel=False)
    assert np.min(img_rescaled.pixels) == 0
    assert np.max(img_rescaled.pixels) == 100
    assert np.all(img_rescaled.pixels[0, 6:, 6:] == 50)
    assert np.all(img_rescaled.pixels[1, 6:, 6:] == 100)


def test_rescale_pixels_per_channel():
    img = Image.init_blank((10, 10), n_channels=2)
    img.pixels[0, 6:, 6:] = 2
    img.pixels[1, 6:, 6:] = 4

    img_rescaled = img.rescale_pixels(0, 100, per_channel=True)
    assert np.min(img_rescaled.pixels) == 0
    assert np.max(img_rescaled.pixels) == 100
    assert np.all(img_rescaled.pixels[0, 6:, 6:] == 100)
    assert np.all(img_rescaled.pixels[1, 6:, 6:] == 100)


def test_rescale_pixels_only_masked():
    img = MaskedImage.init_blank((10, 10), n_channels=1, fill=1)
    img.pixels[0, 0, 0] = 0
    img.pixels[0, 6:, 6:] = 2
    img.mask.pixels[:, 6:, 6:] = False

    img_rescaled = img.rescale_pixels(0, 100)
    assert np.min(img_rescaled.pixels) == 0
    assert np.max(img_rescaled.pixels) == 100
    assert img_rescaled.pixels[0, 0, 0] == 0
    assert img_rescaled.pixels[0, 1, 1] == 100
    assert np.all(img_rescaled.mask.pixels == img.mask.pixels)
