import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pybug.exceptions import DimensionalityError
from nose.tools import raises

from pybug.image.base import BooleanNDImage, MaskedNDImage
from pybug.image import Image


def mask_image_3d_test():
    mask_shape = (120, 121, 13)
    mask_region = np.ones(mask_shape)
    return BooleanNDImage(mask_region)


def test_mask_creation_basics():
    mask_shape = (120, 121, 3)
    mask_region = np.ones(mask_shape)
    mask = BooleanNDImage(mask_region)
    assert_equal(mask.n_channels, 1)
    assert_equal(mask.n_dims, 3)
    assert_equal(mask.shape, mask_shape)


def test_mask_blank():
    mask = BooleanNDImage.blank((56, 12, 3))
    assert(np.all(mask.pixels))


def test_mask_blank_false_fill():
    mask = BooleanNDImage.blank((56, 12, 3), fill=False)
    assert(np.all(~mask.pixels))


def test_mask_n_true_n_false():
    mask = BooleanNDImage.blank((64, 14), fill=False)
    assert_equal(mask.n_true, 0)
    assert_equal(mask.n_false, 64 * 14)
    mask.mask[0, 0] = True
    mask.mask[9, 13] = True
    assert_equal(mask.n_true, 2)
    assert_equal(mask.n_false, 64 * 14 - 2)


def test_mask_true_indices():
    mask = BooleanNDImage.blank((64, 14, 51), fill=False)
    mask.mask[0, 2, 5] = True
    mask.mask[5, 13, 4] = True
    true_indices = mask.true_indices
    true_indices_test = np.array([[0, 2, 5], [5, 13, 4]])
    assert_equal(true_indices, true_indices_test)


def test_mask_false_indices():
    mask = BooleanNDImage.blank((64, 14, 51), fill=True)
    mask.mask[0, 2, 5] = False
    mask.mask[5, 13, 4] = False
    false_indices = mask.false_indices
    false_indices_test = np.array([[0, 2, 5], [5, 13, 4]])
    assert_equal(false_indices, false_indices_test)


def test_mask_true_bounding_extent():
    mask = BooleanNDImage.blank((64, 14, 51), fill=False)
    mask.mask[0, 13, 5] = True
    mask.mask[5, 2, 4] = True
    tbe = mask.true_bounding_extent()
    true_extends = np.array([[0,  5], [2, 13], [4,  5]])
    assert_equal(tbe, true_extends)


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