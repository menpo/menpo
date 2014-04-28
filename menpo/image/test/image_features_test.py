import numpy as np
from numpy.testing import assert_allclose
import menpo.io as pio

from menpo.image import MaskedImage

# Setup the static assets (the breaking_bad image)
rgb_image = pio.import_builtin_asset('breakingbad.jpg')
rgb_image.crop_to_landmarks(boundary=20)
rgb_image.constrain_mask_to_landmarks()


def test_imagewindowiterator_hog_padding():
    n_cases = 10
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(image_height[i, 0],
                                            image_width[i, 0], 1))
        hog = image.features.hog(
            mode='dense', window_step_vertical=window_step_vertical[i, 0],
            window_step_horizontal=window_step_horizontal[i, 0],
            window_step_unit='pixels', padding=True)
        n_windows_horizontal = np.ceil(float(image_width[i, 0]) /
                                       float(window_step_horizontal[i, 0]))
        n_windows_vertical = np.ceil(float(image_height[i, 0]) /
                                     float(window_step_vertical[i, 0]))
        assert_allclose(hog.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_hog_no_padding():
    n_cases = 10
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    window_width = np.random.randint(3, 20, [n_cases, 1])
    window_height = np.random.randint(3, 20, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(image_height[i, 0],
                                            image_width[i, 0], 1))
        hog = image.features.hog(
            mode='dense', cell_size=3, block_size=1,
            window_height=window_height[i, 0], window_width=window_width[i, 0],
            window_unit='pixels',
            window_step_vertical=window_step_vertical[i, 0],
            window_step_horizontal=window_step_horizontal[i, 0],
            window_step_unit='pixels', padding=False)
        n_windows_horizontal = 1 + np.floor(float(image_width[i, 0] -
                                                  window_width[i, 0]) /
                                            float(window_step_horizontal[i, 0]))
        n_windows_vertical = 1 + np.floor(float(image_height[i, 0] -
                                                window_height[i, 0]) /
                                          float(window_step_vertical[i, 0]))
        assert_allclose(hog.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_lbp_padding():
    n_cases = 10
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(image_height[i, 0],
                                            image_width[i, 0], 1))
        lbp = image.features.lbp(
            window_step_vertical=window_step_vertical[i, 0],
            window_step_horizontal=window_step_horizontal[i, 0],
            window_step_unit='pixels', padding=True)
        n_windows_horizontal = np.ceil(float(image_width[i, 0]) /
                                       float(window_step_horizontal[i, 0]))
        n_windows_vertical = np.ceil(float(image_height[i, 0]) /
                                     float(window_step_vertical[i, 0]))
        assert_allclose(lbp.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_lbp_no_padding():
    n_cases = 10
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    radius = np.random.randint(3, 5, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(image_height[i, 0],
                                            image_width[i, 0], 1))
        lbp = image.features.lbp(
            radius=radius[i, 0], samples=8,
            window_step_vertical=window_step_vertical[i, 0],
            window_step_horizontal=window_step_horizontal[i, 0],
            window_step_unit='pixels', padding=False)
        window_size = 2 * radius[i, 0] + 1
        n_windows_horizontal = 1 + np.floor(float(image_width[i, 0] -
                                                  window_size) /
                                            float(window_step_horizontal[i, 0]))
        n_windows_vertical = 1 + np.floor(float(image_height[i, 0] -
                                                window_size) /
                                          float(window_step_vertical[i, 0]))
        assert_allclose(lbp.shape, (n_windows_vertical, n_windows_horizontal))