from pybug import data_path_to
from pybug.features.hog import sparse_hog, dense_hog
from pybug.io import auto_import
import numpy as np
from numpy.testing import assert_allclose
from nose.tools import assert_equals

takeo_path = data_path_to('takeo.ppm')
takeo = auto_import(takeo_path)[0]
greyscale_takeo = takeo.as_greyscale().pixels
takeo = takeo.pixels


def test_rgb_dalaltriggs_sparse_hog():
    descriptors, window_centers, opt_info = sparse_hog(
        takeo, verbose=True, method='dalaltriggs', num_orientations=9,
        cell_size=4, block_size=2,
        gradient_signed=1, l2_norm_clip=0.2)
    assert_allclose(window_centers, [[[113, 75]]])
    assert_equals(opt_info.type, 'sparse')
    assert_equals(opt_info.method, 'dalaltriggs')
    assert_equals(opt_info.horizontal_window_count, 1)
    assert_equals(opt_info.vertical_window_count, 1)
    assert_equals(opt_info.n_windows, 1)
    assert_equals(opt_info.window_height, 225)
    assert_equals(opt_info.window_width, 150)
    assert_equals(opt_info.horizontal_window_step, 75)
    assert_equals(opt_info.vertical_window_step, 112)
    assert_equals(opt_info.horizontal_window_block_count, 36)
    assert_equals(opt_info.vertical_window_block_count, 55)
    assert_equals(opt_info.hog_length_per_block, 36)
    assert_equals(opt_info.hog_length_per_window, 71280)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'rgb')


def test_rgb_zhuramanan_sparse_hog():
    descriptors, window_centers, opt_info = sparse_hog(
        takeo, verbose=True, method='zhuramanan', num_orientations=9,
        cell_size=4, block_size=2,
        gradient_signed=1, l2_norm_clip=0.2)
    assert_allclose(window_centers, [[[113, 75]]])
    assert_equals(opt_info.type, 'sparse')
    assert_equals(opt_info.method, 'zhuramanan')
    assert_equals(opt_info.horizontal_window_count, 1)
    assert_equals(opt_info.vertical_window_count, 1)
    assert_equals(opt_info.n_windows, 1)
    assert_equals(opt_info.window_height, 225)
    assert_equals(opt_info.window_width, 150)
    assert_equals(opt_info.horizontal_window_step, 75)
    assert_equals(opt_info.vertical_window_step, 112)
    assert_equals(opt_info.horizontal_window_block_count, 36)
    assert_equals(opt_info.vertical_window_block_count, 54)
    assert_equals(opt_info.hog_length_per_block, 31)
    assert_equals(opt_info.hog_length_per_window, 60264)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'rgb')


def test_greyscale_dalaltriggs_sparse_hog():
    descriptors, window_centers, opt_info = sparse_hog(
        greyscale_takeo, method='dalaltriggs', verbose=True,
        num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=1, l2_norm_clip=0.2)
    assert_allclose(window_centers, [[[113, 75]]])
    assert_equals(opt_info.type, 'sparse')
    assert_equals(opt_info.method, 'dalaltriggs')
    assert_equals(opt_info.horizontal_window_count, 1)
    assert_equals(opt_info.vertical_window_count, 1)
    assert_equals(opt_info.n_windows, 1)
    assert_equals(opt_info.window_height, 225)
    assert_equals(opt_info.window_width, 150)
    assert_equals(opt_info.horizontal_window_step, 75)
    assert_equals(opt_info.vertical_window_step, 112)
    assert_equals(opt_info.horizontal_window_block_count, 36)
    assert_equals(opt_info.vertical_window_block_count, 55)
    assert_equals(opt_info.hog_length_per_block, 36)
    assert_equals(opt_info.hog_length_per_window, 71280)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'greyscale')


def test_greyscale_zhuramanan_sparse_hog():
    descriptors, window_centers, opt_info = sparse_hog(
        greyscale_takeo, method='zhuramanan', verbose=True,
        num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=1, l2_norm_clip=0.2)
    assert_allclose(window_centers, [[[113, 75]]])
    assert_equals(opt_info.type, 'sparse')
    assert_equals(opt_info.method, 'zhuramanan')
    assert_equals(opt_info.horizontal_window_count, 1)
    assert_equals(opt_info.vertical_window_count, 1)
    assert_equals(opt_info.n_windows, 1)
    assert_equals(opt_info.window_height, 225)
    assert_equals(opt_info.window_width, 150)
    assert_equals(opt_info.horizontal_window_step, 75)
    assert_equals(opt_info.vertical_window_step, 112)
    assert_equals(opt_info.horizontal_window_block_count, 36)
    assert_equals(opt_info.vertical_window_block_count, 54)
    assert_equals(opt_info.hog_length_per_block, 31)
    assert_equals(opt_info.hog_length_per_window, 60264)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'greyscale')


def test_rgb_dalaltriggs_dense_hog():
    descriptors, window_centers, opt_info = dense_hog(
        takeo, method='dalaltriggs', num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=True, l2_norm_clip=0.2,
        window_height=16, window_width=16, window_unit='pixels',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding_enabled=True, verbose=True)
    exp_centers = [a[..., None] for a in np.meshgrid(np.arange(8, 218),
                                                     np.arange(8, 143),
                                                     indexing='ij')]
    exp_centers = np.concatenate(exp_centers, axis=2)
    assert_allclose(window_centers, exp_centers)
    assert_equals(opt_info.type, 'dense')
    assert_equals(opt_info.method, 'dalaltriggs')
    assert_equals(opt_info.horizontal_window_count, 135)
    assert_equals(opt_info.vertical_window_count, 210)
    assert_equals(opt_info.n_windows, 28350)
    assert_equals(opt_info.window_height, 16)
    assert_equals(opt_info.window_width, 16)
    assert_equals(opt_info.horizontal_window_step, 1)
    assert_equals(opt_info.vertical_window_step, 1)
    assert_equals(opt_info.horizontal_window_block_count, 3)
    assert_equals(opt_info.vertical_window_block_count, 3)
    assert_equals(opt_info.hog_length_per_block, 36)
    assert_equals(opt_info.hog_length_per_window, 324)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'rgb')


def test_rgb_zhuramanan_dense_hog():
    descriptors, window_centers, opt_info = dense_hog(
        takeo, method='zhuramanan', num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=True, l2_norm_clip=0.2,
        window_height=16, window_width=16, window_unit='pixels',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding_enabled=True, verbose=True)
    exp_centers = [a[..., None] for a in np.meshgrid(np.arange(8, 218),
                                                     np.arange(8, 143),
                                                     indexing='ij')]
    exp_centers = np.concatenate(exp_centers, axis=2)
    assert_allclose(window_centers, exp_centers)
    assert_equals(opt_info.type, 'dense')
    assert_equals(opt_info.method, 'zhuramanan')
    assert_equals(opt_info.horizontal_window_count, 135)
    assert_equals(opt_info.vertical_window_count, 210)
    assert_equals(opt_info.n_windows, 28350)
    assert_equals(opt_info.window_height, 16)
    assert_equals(opt_info.window_width, 16)
    assert_equals(opt_info.horizontal_window_step, 1)
    assert_equals(opt_info.vertical_window_step, 1)
    assert_equals(opt_info.horizontal_window_block_count, 2)
    assert_equals(opt_info.vertical_window_block_count, 2)
    assert_equals(opt_info.hog_length_per_block, 31)
    assert_equals(opt_info.hog_length_per_window, 124)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'rgb')


def test_greyscale_dalaltriggs_dense_hog():
    descriptors, window_centers, opt_info = dense_hog(
        greyscale_takeo, method='dalaltriggs', num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=True, l2_norm_clip=0.2,
        window_height=16, window_width=16, window_unit='pixels',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding_enabled=True, verbose=True)
    exp_centers = [a[..., None] for a in np.meshgrid(np.arange(8, 218),
                                                     np.arange(8, 143),
                                                     indexing='ij')]
    exp_centers = np.concatenate(exp_centers, axis=2)
    assert_allclose(window_centers, exp_centers)
    assert_equals(opt_info.type, 'dense')
    assert_equals(opt_info.method, 'dalaltriggs')
    assert_equals(opt_info.horizontal_window_count, 135)
    assert_equals(opt_info.vertical_window_count, 210)
    assert_equals(opt_info.n_windows, 28350)
    assert_equals(opt_info.window_height, 16)
    assert_equals(opt_info.window_width, 16)
    assert_equals(opt_info.horizontal_window_step, 1)
    assert_equals(opt_info.vertical_window_step, 1)
    assert_equals(opt_info.horizontal_window_block_count, 3)
    assert_equals(opt_info.vertical_window_block_count, 3)
    assert_equals(opt_info.hog_length_per_block, 36)
    assert_equals(opt_info.hog_length_per_window, 324)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'greyscale')


def test_greyscale_zhuramanan_dense_hog():
    descriptors, window_centers, opt_info = dense_hog(
        greyscale_takeo, method='zhuramanan', num_orientations=9, cell_size=4,
        block_size=2, gradient_signed=True, l2_norm_clip=0.2,
        window_height=16, window_width=16, window_unit='pixels',
        window_step_vertical=1, window_step_horizontal=1,
        window_step_unit='pixels', padding_enabled=True, verbose=True)
    exp_centers = [a[..., None] for a in np.meshgrid(np.arange(8, 218),
                                                     np.arange(8, 143),
                                                     indexing='ij')]
    exp_centers = np.concatenate(exp_centers, axis=2)
    assert_allclose(window_centers, exp_centers)
    assert_equals(opt_info.type, 'dense')
    assert_equals(opt_info.method, 'zhuramanan')
    assert_equals(opt_info.horizontal_window_count, 135)
    assert_equals(opt_info.vertical_window_count, 210)
    assert_equals(opt_info.n_windows, 28350)
    assert_equals(opt_info.window_height, 16)
    assert_equals(opt_info.window_width, 16)
    assert_equals(opt_info.horizontal_window_step, 1)
    assert_equals(opt_info.vertical_window_step, 1)
    assert_equals(opt_info.horizontal_window_block_count, 2)
    assert_equals(opt_info.vertical_window_block_count, 2)
    assert_equals(opt_info.hog_length_per_block, 31)
    assert_equals(opt_info.hog_length_per_window, 124)
    assert_equals(opt_info.image_height, 225)
    assert_equals(opt_info.image_width, 150)
    assert_equals(opt_info.padding_enabled, True)
    assert_equals(opt_info.color, 'greyscale')