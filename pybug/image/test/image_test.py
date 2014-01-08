import numpy as np
from numpy.testing import assert_allclose, assert_equal
from nose.tools import raises

from pybug.image import BooleanNDImage, DepthImage, ShapeImage, MaskedNDImage
from pybug.image import RGBImage, IntensityImage


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
    tbe = mask.bounds_true()
    true_extends_mins = np.array([0, 2, 4])
    true_extends_maxs = np.array([5, 13, 5])
    assert_equal(tbe[0], true_extends_mins)
    assert_equal(tbe[1], true_extends_maxs)


def test_rgb_image_creation():
    pixels = np.ones((120, 120, 3))
    RGBImage(pixels)


def test_intensity_image_creation():
    pixels = np.ones((120, 120))
    IntensityImage(pixels)


def test_2d_crop_without_mask():
    pixels = np.ones((120, 120, 3))
    im = RGBImage(pixels)

    cropped_im = im.cropped_copy([10, 50], [20, 60])

    assert(cropped_im.shape == (10, 10))
    assert(cropped_im.n_channels == 3)
    assert(np.alltrue(cropped_im.shape))


def test_2d_crop_with_mask():
    pixels = np.ones((120, 120, 3))
    mask = np.zeros_like(pixels[..., 0])
    mask[10:100, 20:30] = 1
    im = RGBImage(pixels, mask=mask)
    cropped_im = im.cropped_copy([0, 0], [20, 60])
    assert(cropped_im.shape == (20, 60))
    assert(np.alltrue(cropped_im.shape))


def test_normalize_default():
    pixels = np.ones((120, 120, 3))
    pixels[..., 0] = 0.5
    pixels[..., 1] = 0.2345
    image = RGBImage(pixels)
    image.normalize_inplace()
    assert_allclose(np.mean(image.pixels), 0, atol=1e-10)
    assert_allclose(np.std(image.pixels), 1)


@raises(ValueError)
def test_normalize_no_variance_exception():
    pixels = np.ones((120, 120, 3))
    pixels[..., 0] = 0.5
    pixels[..., 1] = 0.2345
    image = RGBImage(pixels)
    image.normalize_inplace(mode='per_channel')


def test_normalize_per_channel():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    image = MaskedNDImage(pixels)
    image.normalize_inplace(mode='per_channel')
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.std(image.as_vector(keep_channels=True), axis=0), 1)


def test_normalize_masked():
    pixels = np.random.randn(120, 120, 3)
    pixels[..., 1] *= 7
    pixels[..., 0] += -14
    pixels[..., 2] /= 130
    mask = np.zeros((120, 120))
    mask[30:50, 20:30] = 1
    image = MaskedNDImage(pixels, mask=mask)
    image.normalize_inplace(mode='per_channel', limit_to_mask=True)
    assert_allclose(
        np.mean(image.as_vector(keep_channels=True), axis=0), 0, atol=1e-10)
    assert_allclose(
        np.std(image.as_vector(keep_channels=True), axis=0), 1)
