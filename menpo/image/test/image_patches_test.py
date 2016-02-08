import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equals

import menpo.io as mio
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68
from menpo.image.base import (Image, _convert_patches_list_to_single_array,
                              _create_patches_image)
from menpo.shape import PointCloud

#######################
# EXTRACT PATCHES TESTS
#######################
def test_double_type():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert(patches[0].pixels.dtype == np.float64)


def test_float_type():
    image = mio.import_builtin_asset('breakingbad.jpg')
    image.pixels = image.pixels.astype(np.float32)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert(patches[0].pixels.dtype == np.float32)


def test_uint8_type():
    image = mio.import_builtin_asset('breakingbad.jpg', normalise=False)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert(patches[0].pixels.dtype == np.uint8)


def test_uint16_type():
    image = Image.init_blank([100, 100], dtype=np.uint16)
    patch_shape = (16, 16)
    landmarks = PointCloud(np.array([[50, 50.]]))
    patches = image.extract_patches(landmarks,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert(patches[0].pixels.dtype == np.uint16)


def test_int_pointcloud():
    image = Image.init_blank([100, 100])
    patch_shape = (16, 16)
    landmarks = PointCloud(np.array([[50, 50]]))
    patches = image.extract_patches(landmarks,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert(patches[0].pixels.dtype == np.float)


def test_uint8_type_single_array():
    image = mio.import_builtin_asset('breakingbad.jpg', normalise=False)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=True)
    assert(patches.dtype == np.uint8)


def test_squared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert_equals(len(patches), 68)


def test_squared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 15)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert_equals(len(patches), 68)


def test_nonsquared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 18)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert_equals(len(patches), 68)


def test_nonsquared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 17)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert_equals(len(patches), 68)


def test_nonsquared_even_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_shape=patch_shape,
                                    as_single_array=False)
    assert_equals(len(patches), 68)


def test_squared_even_patches_landmarks():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches_around_landmarks('PTS',
                                                     patch_shape=patch_shape,
                                                     as_single_array=False)
    assert_equals(len(patches), 68)


def test_squared_even_patches_single_array():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    as_single_array=True,
                                    patch_shape=patch_shape)
    assert_equals(patches.shape, ((68, 1, 3) + patch_shape))


def test_squared_even_patches_sample_offsets():
    image = mio.import_builtin_asset('breakingbad.jpg')
    sample_offsets = np.array([[0, 0], [1, 0]])
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    sample_offsets=sample_offsets,
                                    as_single_array=False)
    assert_equals(len(patches), 136)


#######################
# SET PATCHES TESTS
#######################
def test_single_ndarray_patch():
    patch_shape = (21, 7)
    n_channels = 4
    im = Image.init_blank(patch_shape, n_channels)
    patch = np.zeros((2, 2, n_channels) + patch_shape)
    patch[1, 0, ...] = np.ones((n_channels,) + patch_shape)
    patch[1, 1, ...] = 2 * np.ones((n_channels,) + patch_shape)
    patch_center = PointCloud(np.array([[10., 3.], [11., 3.]]))
    im.set_patches(patch, patch_center, offset=(0, 0), offset_index=1)
    res = np.zeros(patch_shape)
    res[1:-1, :] = 2
    assert_array_equal(im.pixels[2, ...], res)


def test_single_list_patch():
    patch_shape = (21, 7)
    n_channels = 4
    im = Image.init_blank(patch_shape, n_channels)
    patch = [Image(np.ones((n_channels,) + patch_shape)),
             Image(2 * np.ones((n_channels,) + patch_shape))]
    patch_center = PointCloud(np.array([[10., 3.], [11., 3.]]))
    im.set_patches(patch, patch_center, offset=(0, 0), offset_index=0)
    res = np.ones(patch_shape)
    res[1:-1, :] = 2
    assert_array_equal(im.pixels[2, ...], res)


def test_offset_argument():
    patch_shape = (5, 6)
    offsets = [(0., 0.), [0., 0.], np.array([[1., 1.]]), None]
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_center = PointCloud(np.array([[100., 101.], [50., 41.]]))
    patch = np.zeros((2, 1, image.n_channels) + patch_shape)
    patch[0, 0, ...] = np.ones((image.n_channels,) + patch_shape)
    patch[1, 0, ...] = 2 * np.ones((image.n_channels,) + patch_shape)
    for off in offsets:
        image.set_patches(patch, patch_center, offset=off)
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
    image = mio.import_builtin_asset.lenna_png()
    patches1 = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=True)
    new_image1 = Image.init_blank(image.shape, image.n_channels)
    new_image1.landmarks['LJSON'] = image.landmarks['LJSON']
    new_image1.set_patches_around_landmarks(patches1)
    patches2 = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=False)
    new_image2 = Image.init_blank(image.shape, image.n_channels)
    new_image2.landmarks['LJSON'] = image.landmarks['LJSON']
    new_image2.set_patches_around_landmarks(patches2)
    assert_array_equal(new_image1.pixels, new_image2.pixels)


def test_create_patches_image():
    patch_shape = (7, 14)
    image = mio.import_builtin_asset.lenna_png()
    patches = image.extract_patches_around_landmarks(
        patch_shape=patch_shape, as_single_array=True)
    pc = image.landmarks['LJSON'].lms
    patches_image = _create_patches_image(patches, pc, patches_indices=range(17))
    assert(patches_image.n_channels == patches.shape[2])
    assert(patches_image.landmarks.n_groups == 2)
    assert(patches_image.landmarks['selected_patch_centers'].lms.n_points == 17)
    assert(patches_image.landmarks['all_patch_centers'].lms.n_points == 68)
