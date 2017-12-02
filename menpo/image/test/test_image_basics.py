import warnings

import numpy as np
from numpy.testing import assert_allclose
from pytest import raises
from pathlib import Path

import menpo
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.shape import PointCloud
from menpo.transform import UniformScale, Translation


def test_image_as_masked():
    img = Image(np.random.rand(3, 3, 1), copy=False)
    m_img = img.as_masked()
    assert(type(m_img) == MaskedImage)
    assert_allclose(m_img.pixels, img.pixels)


def test_image_has_nan_values():
    img = Image(np.random.rand(1, 3, 3), copy=False)
    img.pixels[0, 0, 0] = np.nan
    assert img.has_nan_values()


def test_image_no_nan_values():
    img = Image(np.random.rand(1, 3, 3), copy=False)
    assert not img.has_nan_values()


def test_masked_image_as_unmasked():
    m_img = MaskedImage(np.random.rand(1, 3, 3), copy=False)
    img = m_img.as_unmasked()
    assert(type(img) == Image)
    assert_allclose(m_img.pixels, img.pixels)


def test_masked_image_as_unmasked_fill():
    m_img = MaskedImage(np.random.rand(1, 3, 3), copy=False)
    m_img.mask.pixels[0, 0, 0] = False
    img = m_img.as_unmasked(fill=8)
    assert(type(img) == Image)
    assert_allclose(m_img.pixels[0, 1:, 1:], img.pixels[0, 1:, 1:])
    assert_allclose(img.pixels[0, 0, 0], 8.0)


def test_masked_image_as_unmasked_fill_tuple():
    m_img = MaskedImage(np.random.rand(3, 3, 3), copy=False)
    m_img.mask.pixels[0, 0, 0] = False
    img = m_img.as_unmasked(fill=(1, 2, 3))
    assert(type(img) == Image)
    assert_allclose(m_img.pixels[0, 1:, 1:], img.pixels[0, 1:, 1:])
    assert_allclose(img.pixels[:, 0, 0], (1, 2, 3))


def test_boolean_image_as_masked_raises_not_implemented_error():
    b_img = BooleanImage.init_blank((4, 5))
    with raises(NotImplementedError):
        b_img.as_masked()


def test_warp_to_shape_preserves_path():
    bb = menpo.io.import_builtin_asset.breakingbad_jpg()
    bb2 = bb.rescale(0.1)
    assert hasattr(bb2, 'path')
    assert bb2.path == bb.path


def test_warp_to_mask_preserves_path():
    bb = menpo.io.import_builtin_asset.breakingbad_jpg()
    no_op = UniformScale(1.0, n_dims=2)
    bb2 = bb.warp_to_mask(BooleanImage.init_blank((10, 10)), no_op)
    assert hasattr(bb2, 'path')
    assert bb2.path == bb.path


def test_warp_to_shape_boolean_preserves_path():
    i1 = BooleanImage.init_blank((10, 10))
    i1.path = Path('.')
    i2 = i1.rescale(0.8)
    assert hasattr(i2, 'path')
    assert i2.path == i1.path


def test_init_from_rolled_channels():
    p = np.empty([50, 60, 3])
    im = Image.init_from_channels_at_back(p)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60


def test_init_from_channels_at_back_less_dimensions():
    p = np.empty([50, 60])
    im = Image.init_from_channels_at_back(p)
    assert im.n_channels == 1
    assert im.height == 50
    assert im.width == 60


def test_init_from_pointcloud():
    pc = PointCloud.init_2d_grid((10, 10))
    im = Image.init_from_pointcloud(pc)
    assert im.shape == (9, 9)


def test_init_from_pointcloud_return_transform():
    correct_tr = Translation([5, 5])
    pc = correct_tr.apply(PointCloud.init_2d_grid((10, 10)))
    im, tr = Image.init_from_pointcloud(pc, return_transform=True)
    assert im.shape == (9, 9)
    assert_allclose(tr.as_vector(), -correct_tr.as_vector())


def test_init_from_pointcloud_attach_group():
    pc = PointCloud.init_2d_grid((10, 10))
    im = Image.init_from_pointcloud(pc, group='test')
    assert im.shape == (9, 9)
    assert im.n_landmark_groups == 1


def test_init_from_pointcloud_boundary():
    pc = PointCloud.init_2d_grid((10, 10))
    im = Image.init_from_pointcloud(pc, boundary=5)
    print(im.shape)
    assert im.shape == (19, 19)


def test_bounds_2d():
    im = Image.init_blank((50, 30))
    assert_allclose(im.bounds(), ((0, 0), (49, 29)))


def test_bounds_3d():
    im = Image.init_blank((50, 30, 10))
    assert_allclose(im.bounds(), ((0, 0, 0), (49, 29, 9)))


def test_constrain_landmarks_to_bounds():
    im = Image.init_blank((10, 10))
    im.landmarks['test'] = PointCloud.init_2d_grid((20, 20))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        im.constrain_landmarks_to_bounds()
    assert not im.has_landmarks_outside_bounds()
    assert_allclose(im.landmarks['test'].bounds(), im.bounds())
