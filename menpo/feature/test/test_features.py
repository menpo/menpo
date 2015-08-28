from __future__ import division
import random
import numpy as np
from numpy.testing import assert_allclose
from nose.plugins.attrib import attr

from menpo.image import Image, MaskedImage
from menpo.feature import hog, lbp, es, igo, daisy
import menpo.io as mio


def test_imagewindowiterator_hog_padding():
    n_cases = 5
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(1, image_height[i, 0],
                                            image_width[i, 0]))
        hog_im = hog(image, mode='dense',
                     window_step_vertical=window_step_vertical[i, 0],
                     window_step_horizontal=window_step_horizontal[i, 0],
                     window_step_unit='pixels', padding=True)
        n_windows_horizontal = len(range(0, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(0, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(hog_im.shape, (n_windows_vertical,
                                       n_windows_horizontal))


def test_windowiterator_hog_no_padding():
    n_cases = 5
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    window_width = np.random.randint(3, 20, [n_cases, 1])
    window_height = np.random.randint(3, 20, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(1, image_height[i, 0],
                                            image_width[i, 0]))
        hog_img = hog(image, mode='dense', cell_size=3, block_size=1,
                      window_height=window_height[i, 0],
                      window_width=window_width[i, 0], window_unit='pixels',
                      window_step_vertical=window_step_vertical[i, 0],
                      window_step_horizontal=window_step_horizontal[i, 0],
                      window_step_unit='pixels', padding=False)
        n_windows_horizontal = len(range(window_width[i, 0] - 1,
                                         image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(window_height[i, 0] - 1,
                                       image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(hog_img.shape, (n_windows_vertical,
                                        n_windows_horizontal))


def test_windowiterator_lbp_padding():
    n_cases = 5
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(1, image_height[i, 0],
                                            image_width[i, 0]))
        lbp_img = lbp(image, window_step_vertical=window_step_vertical[i, 0],
                      window_step_horizontal=window_step_horizontal[i, 0],
                      window_step_unit='pixels', padding=True)
        n_windows_horizontal = len(range(0, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(0, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(lbp_img.shape, (n_windows_vertical,
                                        n_windows_horizontal))


def test_windowiterator_lbp_no_padding():
    n_cases = 5
    image_width = np.random.randint(50, 250, [n_cases, 1])
    image_height = np.random.randint(50, 250, [n_cases, 1])
    window_step_horizontal = np.random.randint(1, 10, [n_cases, 1])
    window_step_vertical = np.random.randint(1, 10, [n_cases, 1])
    radius = np.random.randint(3, 5, [n_cases, 1])
    for i in range(n_cases):
        image = Image(np.random.randn(1, image_height[i, 0],
                                      image_width[i, 0]))
        lbp_img = lbp(image, radius=radius[i, 0], samples=8,
                      window_step_vertical=window_step_vertical[i, 0],
                      window_step_horizontal=window_step_horizontal[i, 0],
                      window_step_unit='pixels', padding=False)
        window_size = 2 * radius[i, 0] + 1
        n_windows_horizontal = len(range(window_size - 1, image_width[i, 0],
                                         window_step_horizontal[i, 0]))
        n_windows_vertical = len(range(window_size - 1, image_height[i, 0],
                                       window_step_vertical[i, 0]))
        assert_allclose(lbp_img.shape, (n_windows_vertical,
                                        n_windows_horizontal))


def test_hog_channels_dalaltriggs():
    n_cases = 3
    cell_size = np.random.randint(1, 10, [n_cases, 1])
    block_size = np.random.randint(1, 3, [n_cases, 1])
    num_bins = np.random.randint(7, 9, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases, 1])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(channels[i, 0], 40, 40))
        block_size_pixels = cell_size[i, 0] * block_size[i, 0]
        window_width = np.random.randint(block_size_pixels, 40, 1)
        window_height = np.random.randint(block_size_pixels, 40, 1)
        hog_img = hog(image, mode='dense', algorithm='dalaltriggs',
                      cell_size=cell_size[i, 0], block_size=block_size[i, 0],
                      num_bins=num_bins[i, 0], window_height=window_height[0],
                      window_width=window_width[0], window_unit='pixels',
                      window_step_vertical=3, window_step_horizontal=3,
                      window_step_unit='pixels', padding=True)
        length_per_block = block_size[i, 0] * block_size[i, 0] * num_bins[i, 0]
        n_blocks_horizontal = len(range(block_size_pixels - 1, window_width[0],
                                        cell_size[i, 0]))
        n_blocks_vertical = len(range(block_size_pixels - 1, window_height[0],
                                      cell_size[i, 0]))
        n_channels = n_blocks_horizontal * n_blocks_vertical * length_per_block
        assert_allclose(hog_img.n_channels, n_channels)


def test_hog_channels_zhuramanan():
    n_cases = 3
    cell_size = np.random.randint(2, 10, [n_cases])
    channels = np.random.randint(1, 4, [n_cases])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(channels[i], 40, 40))
        win_width = np.random.randint(3 * cell_size[i], 40, 1)
        win_height = np.random.randint(3 * cell_size[i], 40, 1)
        hog_img = hog(image, mode='dense', algorithm='zhuramanan',
                      cell_size=cell_size[i],
                      window_height=win_height[0],
                      window_width=win_width[0],
                      window_unit='pixels', window_step_vertical=3,
                      window_step_horizontal=3,
                      window_step_unit='pixels', padding=True, verbose=True)
        length_per_block = 31
        n_blocks_horizontal = np.floor((win_width[0] / cell_size[i]) + 0.5) - 2
        n_blocks_vertical = np.floor((win_height[0] / cell_size[i]) + 0.5) - 2
        n_channels = n_blocks_horizontal * n_blocks_vertical * length_per_block
        assert_allclose(hog_img.n_channels, n_channels)


@attr('cyvlfeat')
def test_dsift_channels():
    from menpo.feature import dsift
    n_cases = 3
    num_bins_horizontal = np.random.randint(1, 3, [n_cases, 1])
    num_bins_vertical = np.random.randint(1, 3, [n_cases, 1])
    num_or_bins = np.random.randint(7, 9, [n_cases, 1])
    cell_size_horizontal = np.random.randint(1, 10, [n_cases, 1])
    cell_size_vertical = np.random.randint(1, 10, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases])
    for i in range(n_cases):
        image = MaskedImage(np.random.randn(channels[i], 40, 40))
        dsift_img = dsift(image, window_step_horizontal=1,
                          window_step_vertical=1,
                          num_bins_horizontal=num_bins_horizontal[i, 0],
                          num_bins_vertical=num_bins_vertical[i, 0],
                          num_or_bins=num_or_bins[i, 0],
                          cell_size_horizontal=cell_size_horizontal[i, 0],
                          cell_size_vertical=cell_size_vertical[i, 0])
        n_channels = (num_bins_horizontal[i, 0] * num_bins_vertical[i, 0] *
                      num_or_bins[i, 0])
        assert_allclose(dsift_img.n_channels, n_channels)


def test_lbp_channels():
    n_cases = 3
    n_combs = np.random.randint(1, 6, [n_cases, 1])
    channels = np.random.randint(1, 4, [n_cases, 1])
    for i in range(n_cases):
        radius = random.sample(range(1, 10), n_combs[i, 0])
        samples = random.sample(range(4, 12), n_combs[i, 0])
        image = MaskedImage(np.random.randn(channels[i, 0], 40, 40))
        lbp_img = lbp(image, radius=radius, samples=samples,
                      window_step_vertical=3, window_step_horizontal=3,
                      window_step_unit='pixels', padding=True)
        assert_allclose(lbp_img.n_channels, n_combs[i, 0] * channels[i, 0])


def test_igo_channels():
    n_cases = 3
    channels = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = Image(np.random.randn(channels[i, 0], 40, 40))
        igo_img = igo(image)
        igo2_img = igo(image, double_angles=True)
        assert_allclose(igo_img.shape, image.shape)
        assert_allclose(igo2_img.shape, image.shape)
        assert_allclose(igo_img.n_channels, 2 * channels[i, 0])
        assert_allclose(igo2_img.n_channels, 4 * channels[i, 0])


def test_es_channels():
    n_cases = 3
    channels = np.random.randint(1, 10, [n_cases, 1])
    for i in range(n_cases):
        image = Image(np.random.randn(channels[i, 0], 40, 40))
        es_img = es(image)
        assert_allclose(es_img.shape, image.shape)
        assert_allclose(es_img.n_channels, 2 * channels[i, 0])


def test_daisy_channels():
    n_cases = 3
    rings = np.random.randint(1, 3, [n_cases, 1])
    orientations = np.random.randint(1, 7, [n_cases, 1])
    histograms = np.random.randint(1, 6, [n_cases, 1])
    channels = np.random.randint(1, 5, [n_cases, 1])
    for i in range(n_cases):
        image = Image(np.random.randn(channels[i, 0], 40, 40))
        daisy_img = daisy(image, step=4, rings=rings[i, 0],
                          orientations=orientations[i, 0],
                          histograms=histograms[i, 0])
        assert_allclose(daisy_img.shape, (3, 3))
        assert_allclose(daisy_img.n_channels,
                        ((rings[i, 0]*histograms[i, 0]+1)*orientations[i, 0]))


def test_igo_values():
    image = Image([[1., 2.], [2., 1.]])
    igo_img = igo(image)
    res = np.array(
        [[[0.70710678, 0.70710678],
          [-0.70710678, -0.70710678]],
         [[0.70710678, -0.70710678],
          [0.70710678, -0.70710678]]])
    assert_allclose(igo_img.pixels, res)
    image = Image([[0., 0.], [0., 0.]])
    igo_img = igo(image)
    res = np.array([[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]]])
    assert_allclose(igo_img.pixels, res)


def test_es_values():
    image = Image([[1., 2.], [2., 1.]])
    es_img = es(image)
    k = 1.0 / (2 * (2 ** 0.5))
    res = np.array([[[k, -k], [k, -k]], [[k, k], [-k, -k]]])
    assert_allclose(es_img.pixels, res)


def test_daisy_values():
    image = Image([[1., 2., 3., 4.], [2., 1., 3., 4.], [1., 2., 3., 4.],
                   [2., 1., 3., 4.]])
    daisy_img = daisy(image, step=1, rings=2, radius=1, orientations=8,
                      histograms=8)
    assert_allclose(np.around(daisy_img.pixels[10, 0, 0], 6), 0.001355)
    assert_allclose(np.around(daisy_img.pixels[20, 0, 1], 6), 0.032237)
    assert_allclose(np.around(daisy_img.pixels[30, 1, 0], 6), 0.002032)
    assert_allclose(np.around(daisy_img.pixels[40, 1, 1], 6), 0.000163)


@attr('cyvlfeat')
def test_dsift_values():
    from menpo.feature import dsift
    # Equivalent to the transpose of image in Matlab
    image = Image([[1, 2, 3, 4], [2, 1, 3, 4], [1, 2, 3, 4], [2, 1, 3, 4]])
    sift_img = dsift(image, cell_size_horizontal=2, cell_size_vertical=2)
    assert_allclose(np.around(sift_img.pixels[0, 0, 0], 6), 19.719786,
                    rtol=1e-04)
    assert_allclose(np.around(sift_img.pixels[1, 0, 1], 6), 141.535736,
                    rtol=1e-04)
    assert_allclose(np.around(sift_img.pixels[0, 1, 0], 6), 184.377472,
                    rtol=1e-04)
    assert_allclose(np.around(sift_img.pixels[5, 1, 1], 6), 39.04007,
                    rtol=1e-04)
    assert 1

def test_lbp_values():
    image = Image([[0., 6., 0.], [5., 18., 13.], [0., 20., 0.]])
    lbp_img = lbp(image, radius=1, samples=4, mapping_type='none',
                  padding=False)
    assert_allclose(lbp_img.pixels, 8.)
    image = Image([[0., 6., 0.], [5., 25., 13.], [0., 20., 0.]])
    lbp_img = lbp(image, radius=1, samples=4, mapping_type='riu2',
                  padding=False)
    assert_allclose(lbp_img.pixels, 0.)
    image = Image([[0., 6., 0.], [5., 13., 13.], [0., 20., 0.]])
    lbp_img = lbp(image, radius=1, samples=4, mapping_type='u2', padding=False)
    assert_allclose(lbp_img.pixels, 8.)
    image = Image([[0., 6., 0.], [5., 6., 13.], [0., 20., 0.]])
    lbp_img = lbp(image, radius=1, samples=4, mapping_type='ri', padding=False)
    assert_allclose(lbp_img.pixels, 4.)


def test_constrain_landmarks():
    breaking_bad = mio.import_builtin_asset('breakingbad.jpg').as_masked()
    breaking_bad = breaking_bad.crop_to_landmarks(boundary=20)
    breaking_bad = breaking_bad.resize([50, 50])
    breaking_bad.constrain_mask_to_landmarks()
    hog_b = hog(breaking_bad, mode='sparse')
    x = np.where(hog_b.landmarks['PTS'].lms.points[:, 0] > hog_b.shape[1] - 1)
    y = np.where(hog_b.landmarks['PTS'].lms.points[:, 0] > hog_b.shape[0] - 1)
    assert_allclose(len(x[0]) + len(y[0]), 12)
    hog_b = hog(breaking_bad, mode='sparse')
    hog_b.constrain_landmarks_to_bounds()
    x = np.where(hog_b.landmarks['PTS'].lms.points[:, 0] > hog_b.shape[1] - 1)
    y = np.where(hog_b.landmarks['PTS'].lms.points[:, 0] > hog_b.shape[0] - 1)
    assert_allclose(len(x[0]) + len(y[0]), 0)
