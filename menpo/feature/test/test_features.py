from __future__ import division

import warnings

import numpy as np
from numpy.testing import assert_allclose
from pytest import raises, skip

from menpo.feature import (es, igo, daisy, no_op, normalize,
                           normalize_norm, normalize_std, normalize_var)
from menpo.image import Image, MaskedImage
from menpo.testing import is_same_array


def test_dsift_channels():
    try:
        from menpo.feature import dsift
    except ImportError:
        skip("Cyvlfeat must be installed to run this unit test")
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
                        ((rings[i, 0] * histograms[i, 0] + 1) * orientations[i, 0]))


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


def test_dsift_values():
    try:
        from menpo.feature import dsift
    except ImportError:
        skip("Cyvlfeat must be installed to run this unit test")
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


def test_no_op():
    image = Image([[1., 2.], [2., 1.]])
    new_image = no_op(image)
    assert_allclose(image.pixels, new_image.pixels)
    assert not is_same_array(image.pixels, new_image.pixels)


def test_normalize_no_scale_all():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize(image, scale_func=None, mode='all')
    assert_allclose(new_image.pixels, pixels - 13.)


def test_normalize_norm_all():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_norm(image, mode='all')
    assert_allclose(np.linalg.norm(new_image.pixels), 1.)


def test_normalize_norm_channels():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_norm(image, mode='per_channel')
    assert_allclose(np.linalg.norm(new_image.pixels[0]), 1.)
    assert_allclose(np.linalg.norm(new_image.pixels[1]), 1.)
    assert_allclose(np.linalg.norm(new_image.pixels[2]), 1.)


def test_normalize_std_all():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_std(image, mode='all')
    assert_allclose(np.std(new_image.pixels), 1.)


def test_normalize_std_channels():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_std(image, mode='per_channel')
    assert_allclose(np.std(new_image.pixels[0]), 1.)
    assert_allclose(np.std(new_image.pixels[1]), 1.)
    assert_allclose(np.std(new_image.pixels[2]), 1.)


def test_normalize_var_all():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_var(image, mode='all')
    assert_allclose(np.var(new_image.pixels), 0.01648, atol=1e-3)


def test_normalize_var_channels():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize_var(image, mode='per_channel')
    assert_allclose(np.var(new_image.pixels[0]), 0.15, atol=1e-5)
    assert_allclose(np.var(new_image.pixels[1]), 0.15, atol=1e-5)
    assert_allclose(np.var(new_image.pixels[2]), 0.15, atol=1e-5)


def test_normalize_no_scale_per_channel():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize(image, scale_func=None, mode='per_channel')
    assert_allclose(new_image.pixels[0], pixels[0] - 4.)
    assert_allclose(new_image.pixels[1], pixels[1] - 13.)
    assert_allclose(new_image.pixels[2], pixels[2] - 22.)


def test_normalize_no_scale_per_channel():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    new_image = normalize(image, scale_func=None, mode='per_channel')
    assert_allclose(new_image.pixels[0], pixels[0] - 4.)
    assert_allclose(new_image.pixels[1], pixels[1] - 13.)
    assert_allclose(new_image.pixels[2], pixels[2] - 22.)


def test_normalize_scale_all():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    dummy_scale = lambda *a, **kwargs: np.array(2.0)
    image = Image(pixels, copy=False)
    new_image = normalize(image, scale_func=dummy_scale, mode='all')
    assert_allclose(new_image.pixels, (pixels - 13.0) / 2.0)


def test_normalize_scale_per_channel():
    pixels = np.arange(27, dtype=np.float).reshape([3, 3, 3])
    image = Image(pixels, copy=False)
    dummy_scale = lambda *a, **kwargs: np.array(2.0)
    new_image = normalize(image, scale_func=dummy_scale, mode='per_channel')
    assert_allclose(new_image.pixels[0], (pixels[0] - 4.) / 2.0)
    assert_allclose(new_image.pixels[1], (pixels[1] - 13.) / 2.0)
    assert_allclose(new_image.pixels[2], (pixels[2] - 22.) / 2.0)


def test_normalize_unknown_mode_raises():
    image = Image.init_blank((2, 2))
    with raises(ValueError):
        normalize(image, mode='fake')


def test_normalize_0_variance_raises():
    image = Image.init_blank((2, 2))
    dummy_scale = lambda *a, **kwargs: np.array(0.0)
    with raises(ValueError):
        normalize(image, scale_func=dummy_scale)


def test_normalize_0_variance_warning():
    pixels = np.arange(8, dtype=np.float).reshape([2, 2, 2])
    image = Image(pixels, copy=False)
    dummy_scale = lambda *a, **kwargs: np.array([2.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        new_image = normalize(image, scale_func=dummy_scale,
                              error_on_divide_by_zero=False,
                              mode='per_channel')
    assert_allclose(new_image.pixels[0], [[-0.75, -0.25], [0.25, 0.75]])
    assert_allclose(new_image.pixels[1], [[-1.5, -0.5], [0.5, 1.5]])
