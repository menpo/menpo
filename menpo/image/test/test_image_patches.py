import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

import menpo.io as mio
from menpo.image.base import (Image, _convert_patches_list_to_single_array,
                              _create_patches_image)
from menpo.image.patches import (extract_patches_with_slice,
                                 extract_patches_by_sampling)
from menpo.shape import PointCloud


#######################
# EXTRACT PATCHES TESTS
#######################
def test_double_type():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert (patches[0].pixels.dtype == np.float64)


def test_float_type():
    image = mio.import_builtin_asset('breakingbad.jpg')
    image.pixels = image.pixels.astype(np.float32)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert (patches[0].pixels.dtype == np.float32)


def test_uint8_type():
    image = mio.import_builtin_asset('breakingbad.jpg', normalize=False)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert (patches[0].pixels.dtype == np.uint8)


def test_uint16_type():
    image = Image.init_blank([100, 100], dtype=np.uint16)
    patch_shape = (16, 16)
    landmarks = PointCloud(np.array([[50, 50.]]))
    patches = image.extract_patches(landmarks,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert (patches[0].pixels.dtype == np.uint16)


def test_int_pointcloud():
    image = Image.init_blank([100, 100])
    patch_shape = (16, 16)
    landmarks = PointCloud(np.array([[50, 50]]))
    patches = image.extract_patches(landmarks,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert (patches[0].pixels.dtype == np.float)


def test_uint8_type_single_array():
    image = mio.import_builtin_asset('breakingbad.jpg', normalize=False)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=True)
    assert (patches.dtype == np.uint8)


def test_squared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert len(patches) == 68


def test_squared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 15)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert len(patches) == 68


def test_nonsquared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 18)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert len(patches) == 68


def test_nonsquared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 17)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert len(patches) == 68


def test_nonsquared_even_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert len(patches) == 68


def test_squared_even_patches_landmarks():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches_around_landmarks('PTS',
                                                     patch_shape=patch_shape,
                                                     as_single_array=False)
    assert len(patches) == 68


def test_squared_even_patches_single_array():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'],
                                    as_single_array=True,
                                    patch_shape=patch_shape)
    assert patches.shape, ((68, 1 == 3) + patch_shape)


def test_squared_even_patches_sample_offsets():
    image = mio.import_builtin_asset('breakingbad.jpg')
    sample_offsets = np.array([[0, 0], [1, 0]])
    patches = image.extract_patches(image.landmarks['PTS'],
                                    sample_offsets=sample_offsets,
                                    as_single_array=False)
    assert len(patches) == 136


@pytest.mark.parametrize('patch_shape,expected_valid', [((15, 13), (8, 7)),
                                                        ((16, 12), (8, 6))],
                         ids=str)
def test_slicing_out_of_bounds(patch_shape, expected_valid):
    image = mio.import_builtin_asset('breakingbad.jpg')
    sample_offsets = np.array([[0, 0], [1, 0]])
    cval = -100
    points = np.array([[0, 0.]])
    sliced_patches = extract_patches_with_slice(
        image.pixels, points, patch_shape, sample_offsets, cval=cval)

    offset = (patch_shape[0] // 2, patch_shape[1] // 2)
    assert_allclose(sliced_patches[0, 0, :, offset[0]:, offset[1]:],
                    image.pixels[:, :expected_valid[0], :expected_valid[1]],
                    rtol=1e-4)
    assert_allclose(sliced_patches[0, 0, :, :offset[0], :offset[1]], -100)


@pytest.mark.parametrize('patch_shape,expected_valid', [((15, 13), (8, 7)),
                                                        ((16, 12), (8, 6))],
                         ids=str)
def test_sampling_out_of_bounds(patch_shape, expected_valid):
    image = mio.import_builtin_asset('breakingbad.jpg')
    sample_offsets = np.array([[0, 0], [1, 0]])
    cval = -100
    points = np.array([[0, 0.]])
    sliced_patches = extract_patches_by_sampling(
        image.pixels, points, patch_shape, sample_offsets, cval=cval)

    offset = (patch_shape[0] // 2, patch_shape[1] // 2)
    assert_allclose(sliced_patches[0, 0, :, offset[0]:, offset[1]:],
                    image.pixels[:, :expected_valid[0], :expected_valid[1]],
                    rtol=1e-4)
    assert_allclose(sliced_patches[0, 0, :, :offset[0], :offset[1]], -100)

    # Offset in row direction by 1
    assert_allclose(sliced_patches[0, 1, :, offset[0]:, offset[1]:],
                    image.pixels[:, 1:expected_valid[0] + 1, :expected_valid[1]],
                    rtol=1e-4)
    assert_allclose(sliced_patches[0, 1, :, :offset[0], :offset[1]], -100)


@pytest.mark.parametrize('patch_shape', [(15, 13), (16, 12)], ids=str)
def test_slicing_equals_sampling(patch_shape):
    image = mio.import_builtin_asset('breakingbad.jpg')
    sample_offsets = np.array([[0, 0], [1, 0]])
    points = image.landmarks['PTS'].points
    cval = -100
    # Add an extra point that is partially out of bounds
    points = np.concatenate([points, [[0, 0]]])
    sliced_patches = extract_patches_with_slice(
        image.pixels, points, patch_shape, sample_offsets, cval=cval)
    sampled_patches = extract_patches_by_sampling(
        image.pixels, points, patch_shape, sample_offsets, order=0, cval=cval)

    assert_allclose(sliced_patches, sampled_patches, rtol=1e-4)
    assert_allclose(sliced_patches[-1, 0, 0, 0, 0], -100)


#######################
# SET PATCHES TESTS
#######################
def test_single_ndarray_patch():
    patch_shape = (8, 7)
    n_channels = 4
    im = Image.init_blank((32, 32), n_channels)
    patch = np.zeros((2, 2, n_channels) + patch_shape)
    patch[0, 0, ...] = np.full((n_channels,) + patch_shape, 1)  # Should be unused
    patch[0, 1, ...] = np.full((n_channels,) + patch_shape, 2)
    patch[1, 0, ...] = np.full((n_channels,) + patch_shape, 3)  # Should be unused
    patch[1, 1, ...] = np.full((n_channels,) + patch_shape, 4)
    patch_center = PointCloud(np.array([[4., 4.], [16., 16.]]))
    new_im = im.set_patches(patch, patch_center, offset_index=1)
    res = np.zeros((32, 32))
    res[:8, 1:8] = 2
    res[12:20, 13:20] = 4
    assert_array_equal(new_im.pixels[2], res)


def test_single_list_patch():
    patch_shape = (8, 7)
    n_channels = 4
    im = Image.init_blank((32, 32), n_channels)
    patch = [Image(np.full((n_channels,) + patch_shape, 1)),
             Image(np.full((n_channels,) + patch_shape, 2)),  # Should be unused
             Image(np.full((n_channels,) + patch_shape, 3)),
             Image(np.full((n_channels,) + patch_shape, 4))]  # Should be unused
    patch_center = PointCloud(np.array([[4., 4.], [16., 16.]]))
    new_im = im.set_patches(patch, patch_center, offset_index=0)
    res = np.zeros((32, 32))
    res[:8, 1:8] = 1
    res[12:20, 13:20] = 3
    assert_array_equal(new_im.pixels[0], res)


def test_offset_argument():
    patch_shape = (5, 6)
    offsets = [(0., 0.), [0., 0.], np.array([[1., 1.]]), None]
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_center = PointCloud(np.array([[100., 101.], [50., 41.]]))
    patch = np.zeros((2, 1, image.n_channels) + patch_shape)
    patch[0, 0, ...] = np.ones((image.n_channels,) + patch_shape)
    patch[1, 0, ...] = 2 * np.ones((image.n_channels,) + patch_shape)
    for off in offsets:
        image = image.set_patches(patch, patch_center, offset=off)
        assert_array_equal(image.pixels[:, 98:103, 98:104], patch[0, 0, ...])
        assert_array_equal(image.pixels[:, 48:53, 38:44], patch[1, 0, ...])


def test_convert_patches_list_to_single_array():
    patch_shape = (7, 2)
    n_channels = 10
    n_centers = 2
    n_offsets = 2
    patches_list = [Image(1 * np.ones((n_channels,) + patch_shape)),
                    Image(2 * np.ones((n_channels,) + patch_shape)),
                    Image(3 * np.ones((n_channels,) + patch_shape)),
                    Image(4 * np.ones((n_channels,) + patch_shape))]
    patches_array = np.zeros((n_centers, n_offsets, n_channels) + patch_shape)
    patches_array[0, 0, ...] = patches_list[0].pixels
    patches_array[0, 1, ...] = patches_list[1].pixels
    patches_array[1, 0, ...] = patches_list[2].pixels
    patches_array[1, 1, ...] = patches_list[3].pixels
    assert_array_equal(
        _convert_patches_list_to_single_array(patches_list, n_centers),
        patches_array)


def test_set_patches_around_landmarks():
    patch_shape = (21, 12)
    image = mio.import_builtin_asset.takeo_ppm()
    patches1 = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=True)
    new_image1 = Image.init_blank(image.shape, image.n_channels)
    new_image1.landmarks['PTS'] = image.landmarks['PTS']
    extracted1 = new_image1.set_patches_around_landmarks(patches1)

    patches2 = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=False)
    new_image2 = Image.init_blank(image.shape, image.n_channels)
    new_image2.landmarks['PTS'] = image.landmarks['PTS']
    extracted2 = new_image2.set_patches_around_landmarks(patches2)
    assert_array_equal(extracted1.pixels, extracted2.pixels)


def test_create_patches_image():
    patch_shape = (7, 14)
    image = mio.import_builtin_asset.takeo_ppm()
    patches = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=True)
    pc = image.landmarks['PTS']
    patches_image = _create_patches_image(patches, pc,
                                          patches_indices=list(range(17)))
    assert (patches_image.n_channels == patches.shape[2])
    assert (patches_image.landmarks.n_groups == 1)
    assert (patches_image.landmarks['patch_centers'].n_points == 17)
