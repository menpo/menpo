import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from pathlib import Path

import menpo
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.transform import UniformScale


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


@raises(NotImplementedError)
def test_boolean_image_as_masked_raises_not_implemented_error():
    b_img = BooleanImage.init_blank((4, 5))
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
    im = Image.init_from_rolled_channels(p)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60
