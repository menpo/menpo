import numpy as np
from numpy.testing import assert_allclose
from pybug.exceptions import DimensionalityError
from pybug.image import Image
from nose.tools import raises


def test_image_creation():
    pixels = np.ones((120, 120, 3))
    Image(pixels)


def test_2d_crop_without_mask():
    pixels = np.ones((120, 120, 3))
    im = Image(pixels)

    cropped_im, translation = im.crop(slice(10, 20), slice(50, 60))

    assert(cropped_im.shape == (10, 10))
    assert(cropped_im.n_channels == 3)
    assert(np.alltrue(cropped_im.shape))
    assert_allclose(translation.as_vector(), [10, 50])


def test_2d_crop_with_mask():
    pixels = np.ones((120, 120, 3))
    mask = np.zeros_like(pixels[..., 0])
    mask[10:100, 20:30] = 1
    im = Image(pixels, mask=mask)

    cropped_im, translation = im.crop(slice(0, 20), slice(0, 60))

    assert(cropped_im.shape == (20, 60))
    assert(np.alltrue(cropped_im.shape))

    correct_mask = np.zeros([20, 60])
    correct_mask[10:, 20:30] = 1
    assert_allclose(cropped_im.mask.mask, correct_mask)
    assert_allclose(translation.as_vector(), [0, 0])


@raises(AssertionError)
def test_crop_wrong_arg_num_raises_assertionerror():
    pixels = np.ones((120, 120, 3))
    im = Image(pixels)

    im.crop(slice(0, 20))