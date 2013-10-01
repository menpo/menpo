import numpy as np
from numpy.testing import assert_allclose
from pybug.interpolation.cinterp import interp2
from pybug.io import auto_import
from pybug import data_path_to


# Setup the static assets (the takeo image)
takeo_path = data_path_to('takeo.ppm')
base_image = auto_import(takeo_path)[0]
gray_image = base_image.as_greyscale()

gray_template = gray_image.cropped_copy([70, 30], [169, 129])
multi_template = base_image.cropped_copy([70, 30], [169, 129])
initial_params = np.array([0, 0, 0, 0, 70, 30])
row_indices, col_indices = np.meshgrid(np.arange(50, 100), np.arange(50, 100),
                                       indexing='ij')
row_indices, col_indices = row_indices.flatten(), col_indices.flatten()
multi_expected = base_image.cropped_copy([50, 50],
                                         [100, 100]).pixels.flatten()


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
