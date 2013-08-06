import numpy as np
from numpy.testing import assert_allclose
from pybug.transform import AffineTransform
from pybug.warp import scipy_warp, cinterp2_warp
from pybug.warp.cinterp import interp2
from pybug.io import auto_import
from pybug import data_path_to


# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
base_image = auto_import(takeo_path)[0]
gray_image = base_image.as_greyscale()

gray_template, gray_translation = gray_image.crop(slice(70, 169), slice(30, 129))
multi_template, multi_translation = base_image.crop(slice(70, 169), slice(30, 129))
initial_params = np.array([0, 0, 0, 0, 70, 30])
row_indices, col_indices = np.meshgrid(np.arange(50, 100), np.arange(50, 100),
                                       indexing='ij')
row_indices, col_indices = row_indices.flatten(), col_indices.flatten()
multi_expected = base_image.crop(slice(50, 100), slice(50, 100))[0].pixels.flatten()


def test_scipy_warp_gray():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = scipy_warp(gray_image, gray_template, target_transform)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_scipy_warp_multi():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = scipy_warp(base_image, multi_template, target_transform)

    assert(warped_im.shape == multi_template.shape)
    assert_allclose(warped_im.pixels, multi_template.pixels)


# TODO: Not 100% on the best way to test this?
def test_scipy_warp_gray_warp_mask():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = scipy_warp(gray_image, gray_template, target_transform,
                           warp_mask=True)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_cinterp_nearest():
    interp_pixels = interp2(base_image.pixels, row_indices,
                            col_indices, mode='nearest')
    interp_pixels = np.reshape(interp_pixels, [50, 50, 3])

    assert_allclose(interp_pixels.flatten(), multi_expected)


def test_cinterp_bilinear():
    interp_pixels = interp2(base_image.pixels, row_indices,
                            col_indices, mode='bilinear')
    interp_pixels = np.reshape(interp_pixels, [50, 50, 3])

    assert_allclose(interp_pixels.flatten(), multi_expected)


def test_cinterp_bicubic():
    interp_pixels = interp2(base_image.pixels, row_indices,
                            col_indices, mode='bicubic')
    interp_pixels = np.reshape(interp_pixels, [50, 50, 3])

    assert_allclose(interp_pixels.flatten(), multi_expected)


def test_cinterp2_warp_gray():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = cinterp2_warp(gray_image, gray_template, target_transform)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_cinterp2_warp_multi():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = cinterp2_warp(base_image, multi_template, target_transform)

    assert(warped_im.shape == multi_template.shape)
    assert_allclose(warped_im.pixels, multi_template.pixels)


# TODO: Not 100% on the best way to test this?
def test_cinterp2_warp_gray_warp_mask():
    target_transform = AffineTransform.from_vector(initial_params)
    warped_im = cinterp2_warp(gray_image, gray_template, target_transform,
                              warp_mask=True)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)