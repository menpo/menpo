import numpy as np
from nose.tools import raises
from pathlib import Path

import menpo
from menpo.image import Image, MaskedImage, BooleanImage
from menpo.transform import UniformScale


def test_image_as_masked():
    img = Image(np.random.rand(3, 3, 1), copy=False)
    m_img = img.as_masked()
    assert(type(m_img) == MaskedImage)
    assert(np.all(m_img.pixels == img.pixels))


def test_masked_image_as_unmasked():
    m_img = MaskedImage(np.random.rand(3, 3, 1), copy=False)
    img = m_img.as_unmasked()
    assert(type(img) == Image)
    assert(np.all(m_img.pixels == img.pixels))


@raises(NotImplementedError)
def test_boolean_image_as_masked_raises_not_implemented_error():
    b_img = BooleanImage.blank((4, 5))
    b_img.as_masked()


def test_warp_to_shape_preserves_path():
    bb = menpo.io.import_builtin_asset.breakingbad_jpg()
    bb2 = bb.rescale(0.1)
    assert hasattr(bb2, 'path')
    assert bb2.path == bb.path


def test_warp_to_mask_preserves_path():
    bb = menpo.io.import_builtin_asset.breakingbad_jpg()
    no_op = UniformScale(1.0, n_dims=2)
    bb2 = bb.warp_to_mask(BooleanImage.blank((10, 10)), no_op)
    assert hasattr(bb2, 'path')
    assert bb2.path == bb.path


def test_warp_to_shape_boolean_preserves_path():
    i1 = BooleanImage.blank((10, 10))
    i1.path = Path('.')
    i2 = i1.rescale(0.8)
    assert hasattr(i2, 'path')
    assert i2.path == i1.path
