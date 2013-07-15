import numpy as np
from numpy.testing import assert_allclose
from pybug.image import Image
from nose.tools import raises


def test_image_creation():
    pixels = np.ones((120, 120, 3))
    Image(pixels)


def test_2d_crop_without_mask():
    pixels = np.ones((120, 120, 3))
    im = Image(pixels)

    cropped_im, translation = im.crop(slice(10, 20), slice(50, 60))

    assert(cropped_im.image_shape == (10, 10))
    assert(cropped_im.n_channels == 3)
    assert(np.alltrue(cropped_im.image_shape))
    assert_allclose(translation.as_vector(), [0, 0, 0, 0, 10, 50])


def test_2d_crop_with_mask():
    pixels = np.ones((120, 120, 3))
    mask = np.zeros_like(pixels[..., 0])
    mask[10:100, 20:30] = 1
    im = Image(pixels, mask=mask)

    cropped_im, translation = im.crop(slice(0, 20), slice(0, 60))

    assert(cropped_im.image_shape == (20, 60))
    assert(np.alltrue(cropped_im.image_shape))

    correct_mask = np.zeros([20, 60])
    correct_mask[10:, 20:30] = 1
    assert_allclose(cropped_im.mask, correct_mask)
    assert_allclose(translation.as_vector(), [0, 0, 0, 0, 0, 0])


def test_3d_crop_without_mask():
    pixels = np.ones((120, 120, 120, 3))
    im = Image(pixels)

    cropped_im, translation = im.crop(slice(10, 20), slice(50, 60),
                                      slice(100, 120))

    assert(cropped_im.image_shape == (10, 10, 20))
    assert(cropped_im.n_channels == 3)
    assert(np.alltrue(cropped_im.image_shape))
    assert_allclose(translation.as_vector(),
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 50, 100])


def test_3d_crop_with_mask():
    pixels = np.ones((120, 120, 120, 3))
    mask = np.zeros_like(pixels[..., 0])
    mask[10:100, 20:30, 0:10] = 1
    im = Image(pixels, mask=mask)

    cropped_im, translation = im.crop(slice(0, 20), slice(0, 60), slice(0, 10))

    assert(cropped_im.image_shape == (20, 60, 10))
    assert(np.alltrue(cropped_im.image_shape))

    correct_mask = np.zeros([20, 60, 10])
    correct_mask[10:, 20:30, :10] = 1
    assert_allclose(cropped_im.mask, correct_mask)
    assert_allclose(translation.as_vector(),
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


@raises(ValueError)
def test_4d_crop_raises_valueerror():
    pixels = np.ones((120, 120, 120, 120, 3))
    im = Image(pixels)

    im.crop(slice(0, 20), slice(0, 60), slice(0, 10), slice(0, 10))


@raises(AssertionError)
def test_crop_wrong_arg_num_raises_assertionerror():
    pixels = np.ones((120, 120, 3))
    im = Image(pixels)

    im.crop(slice(0, 20))