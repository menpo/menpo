import numpy as np
from nose.tools import raises
from numpy.testing import assert_allclose
from menpo.image import Image, MaskedImage, BooleanImage


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
