import menpo.io as mio

from menpo.fitmultilevel.functions import extract_local_patches_fast


def test_squared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 16)
    patches = extract_local_patches_fast(
        image, image.landmarks['PTS'].lms, patch_shape)
    assert(patches.shape == (68,) + patch_shape + (3,))


def test_squared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 15)
    patches = extract_local_patches_fast(
        image, image.landmarks['PTS'].lms, patch_shape)
    assert(patches.shape == (68,) + patch_shape + (3,))


def test_nonsquared_even_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (16, 18)
    patches = extract_local_patches_fast(
        image, image.landmarks['PTS'].lms, patch_shape)
    assert(patches.shape == (68,) + patch_shape + (3,))


def test_nonsquared_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 17)
    patches = extract_local_patches_fast(
        image, image.landmarks['PTS'].lms, patch_shape)
    assert(patches.shape == (68,) + patch_shape + (3,))


def test_nonsquared_even_odd_patches():
    image = mio.import_builtin_asset('breakingbad.jpg')
    patch_shape = (15, 16)
    patches = extract_local_patches_fast(
        image, image.landmarks['PTS'].lms, patch_shape)
    assert(patches.shape == (68,) + patch_shape + (3,))
