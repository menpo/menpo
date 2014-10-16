from nose.tools import assert_equals

import menpo.io as mio
from menpo.landmark import labeller, ibug_face_68
from menpo.shape import PointCloud


def test_squared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 15)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 18)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 17)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_nonsquared_even_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_even_patches_landmarks():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = image.extract_patches_around_landmarks('PTS',
                                                     patch_size=patch_shape)
    assert_equals(len(patches), 68)


def test_squared_even_patches_landmarks_label():
    image = mio.import_builtin_asset('breakingbad.jpg')
    image = labeller(image, 'PTS', ibug_face_68)
    patch_shape = (16, 16)
    patches = image.extract_patches_around_landmarks('ibug_face_68',
                                                     label='nose',
                                                     patch_size=patch_shape)
    assert_equals(len(patches), 9)


def test_squared_even_patches_single_array():
    image = mio.import_builtin_asset('breakingbad.jpg')
    image = labeller(image, 'PTS', ibug_face_68)
    patch_shape = (16, 16)
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    as_single_array=True,
                                    patch_size=patch_shape)
    assert_equals(patches.shape, ((68,) + patch_shape + (3,)))


def test_squared_even_patches_sample_offsets():
    image = mio.import_builtin_asset('breakingbad.jpg')
    image = labeller(image, 'PTS', ibug_face_68)
    sample_offsets = PointCloud([[0, 0], [1, 0]])
    patches = image.extract_patches(image.landmarks['PTS'].lms,
                                    sample_offsets=sample_offsets)
    assert_equals(len(patches), 136)
