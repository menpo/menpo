import numpy as np
from numpy.testing import assert_allclose
import menpo.io as pio
import random
import math

from menpo.image import MaskedImage

# Setup the static assets (the breaking_bad image)
rgb_image = pio.import_builtin_asset('breakingbad.jpg')
rgb_image.crop_to_landmarks(boundary=20)
rgb_image.constrain_mask_to_landmarks()


def test_imagewindowiterator_hog_padding():
    n_cases = 5
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
        n_windows_horizontal = len(range(0, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(0, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(hog.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_hog_no_padding():
    n_cases = 5
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
        n_windows_horizontal = len(range(window_width[i, 0] - 1,
                                         image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(window_height[i, 0] - 1,
                                       image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(hog.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_lbp_padding():
    n_cases = 5
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
        n_windows_horizontal = len(range(0, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(0, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(lbp.shape, (n_windows_vertical, n_windows_horizontal))


def test_imagewindowiterator_lbp_no_padding():
    n_cases = 5
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
        n_windows_horizontal = len(range(window_size - 1, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(window_size - 1, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(lbp.shape, (n_windows_vertical, n_windows_horizontal))


def test_hog_channels_dalaltriggs():
    n_cases = 3
    cell_size = np.random.randint(1, 10, [n_cases, 1])
    block_size = np.random.randint(1, 3, [n_cases, 1])
    num_bins = np.random.randint(7, 9, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(40, 40, channels[i, 0]))
        block_size_pixels = cell_size[i, 0] * block_size[i, 0]
        window_width = np.random.randint(block_size_pixels, 40, 1)
        window_height = np.random.randint(block_size_pixels, 40, 1)
        hog = image.features.hog(mode='dense', algorithm='dalaltriggs',
                                 cell_size=cell_size[i, 0],
                                 block_size=block_size[i, 0],
                                 num_bins=num_bins[i, 0],
                                 window_height=window_height[0],
                                 window_width=window_width[0],
                                 window_unit='pixels', window_step_vertical=3,
                                 window_step_horizontal=3,
                                 window_step_unit='pixels', padding=True)
        length_per_block = block_size[i, 0] * block_size[i, 0] * num_bins[i, 0]
        n_blocks_horizontal = len(range(block_size_pixels - 1, window_width[0],
                                        cell_size[i, 0]))
        n_blocks_vertical = len(range(block_size_pixels - 1, window_height[0],
                                      cell_size[i, 0]))
        n_channels = n_blocks_horizontal * n_blocks_vertical * length_per_block
        assert_allclose(hog.n_channels, n_channels)


def test_hog_channels_zhuramanan():
    n_cases = 3
    cell_size = np.random.randint(2, 10, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(40, 40, channels[i, 0]))
        window_width = np.random.randint(3 * cell_size[i, 0], 40, 1)
        window_height = np.random.randint(3 * cell_size[i, 0], 40, 1)
        hog = image.features.hog(mode='dense', algorithm='zhuramanan',
                                 cell_size=cell_size[i, 0],
                                 window_height=window_height[0],
                                 window_width=window_width[0],
                                 window_unit='pixels', window_step_vertical=3,
                                 window_step_horizontal=3,
                                 window_step_unit='pixels', padding=True)
        length_per_block = 31
        n_blocks_horizontal = np.round(np.float(window_width[0])
                                       / np.float(cell_size[i, 0])) - 2
        n_blocks_vertical = np.round(np.float(window_height[0])
                                     / np.float(cell_size[i, 0])) - 2
        n_channels = n_blocks_horizontal * n_blocks_vertical * length_per_block
        assert_allclose(hog.n_channels, n_channels)


def test_lbp_channels():
    n_cases = 3
    n_combs = np.random.randint(1, 6, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases, 1])
    for i in range(n_cases):
        radius = random.sample(xrange(1, 10), n_combs[i, 0])
        samples = random.sample(xrange(4, 12), n_combs[i, 0])
        image = MaskedImage(np.random.randn(40, 40, channels[i, 0]))
        lbp = image.features.lbp(radius=radius, samples=samples,
                                 window_step_vertical=3,
                                 window_step_horizontal=3,
                                 window_step_unit='pixels', padding=True)
        assert_allclose(lbp.n_channels, n_combs[i, 0] * channels[i, 0])


def test_igo_channels():
    n_cases = 3
    channels = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(40, 40, channels[i, 0]))
        igo = image.features.igo()
        igo2 = image.features.igo(double_angles=True)
        assert_allclose(igo.shape, image.shape)
        assert_allclose(igo2.shape, image.shape)
        assert_allclose(igo.n_channels, 2 * channels[i, 0])
        assert_allclose(igo2.n_channels, 4 * channels[i, 0])


def test_es_channels():
    n_cases = 3
    channels = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(40, 40, channels[i, 0]))
        es = image.features.es()
        assert_allclose(es.shape, image.shape)
        assert_allclose(es.n_channels, 2 * channels[i, 0])


def test_igo_values():
    image = MaskedImage([[1, 2], [2, 1]])
    igo = image.features.igo()
    res = np.array([
        [[math.cos(math.radians(45)), math.sin(math.radians(45))],
         [math.cos(math.radians(90+45)), math.sin(math.radians(90+45))]],
        [[math.cos(math.radians(-45)), math.sin(math.radians(-45))],
         [math.cos(math.radians(180+45)), math.sin(math.radians(180+45))]]])
    assert_allclose(igo.pixels, res)
    image = MaskedImage([[0, 0], [0, 0]])
    igo = image.features.igo()
    res = np.array([[[1., 0.], [1., 0.]], [[1., 0.], [1., 0.]]])
    assert_allclose(igo.pixels, res)


def test_es_values():
    image = MaskedImage([[1, 2], [2, 1]])
    es = image.features.es()
    k = 1 / (2 * (2**0.5))
    res = np.array([[[k, k], [-k, k]], [[k, -k], [-k, -k]]])
    assert_allclose(es.pixels, res)
    image = MaskedImage([[0, 0], [0, 0]])
    es = image.features.es()
    res = np.array([[[np.nan, np.nan], [np.nan, np.nan]],
                    [[np.nan, np.nan], [np.nan, np.nan]]])
    assert_allclose(es.pixels, res)