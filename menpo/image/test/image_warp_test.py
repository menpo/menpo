import numpy as np
from numpy.testing import assert_allclose
from menpo.transform import Affine
import menpo.io as pio


# Setup the static assets (the takeo image)
rgb_image = pio.import_builtin_asset('takeo.ppm')
gray_image = rgb_image.as_greyscale()

gray_template = gray_image.crop(np.array([70, 30]),
                                        np.array([169, 129]))
rgb_template = rgb_image.crop(np.array([70, 30]),
                                      np.array([169, 129]))
template_mask = gray_template.mask

initial_params = np.array([0, 0, 0, 0, 70, 30])
row_indices, col_indices = np.meshgrid(np.arange(50, 100), np.arange(50, 100),
                                       indexing='ij')
row_indices, col_indices = row_indices.flatten(), col_indices.flatten()
multi_expected = rgb_image.crop([50, 50],
                                        [100, 100]).pixels.flatten()


def test_scipy_warp_gray():
    target_transform = Affine.identity(2).from_vector(initial_params)
    warped_im = gray_image.warp_to(template_mask, target_transform)

    assert(warped_im.shape == gray_template.shape)
    assert_allclose(warped_im.pixels, gray_template.pixels)


def test_scipy_warp_multi():
    target_transform = Affine.identity(2).from_vector(initial_params)
    warped_im = rgb_image.warp_to(template_mask, target_transform)

    assert(warped_im.shape == rgb_template.shape)
    assert_allclose(warped_im.pixels, rgb_template.pixels)
