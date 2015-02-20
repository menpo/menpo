from nose.tools import raises
import numpy as np
from numpy.testing import assert_allclose
from menpo.image import Image
from menpo.feature import gradient

from menpo.feature.features import _np_gradient
import menpo.io as mio


takeo = mio.import_builtin_asset.takeo_ppm()
example_image = np.array([[1., 2., 6.], [3., 4., 5.]])
y_grad = np.array([[2., 2., -1.], [2., 2., -1.]])
x_grad = np.array([[1., 2.5, 4.], [1., 1., 1.]])


def test_gradient_float():
    dtype = np.float32
    p = example_image.astype(dtype)
    image = Image(p)
    grad_image = gradient(image)
    _check_assertions(grad_image, image.shape, image.n_channels * 2,
                      dtype)
    np_grad = np.gradient(p)
    assert_allclose(grad_image.pixels[0], np_grad[0])
    assert_allclose(grad_image.pixels[1], np_grad[1])


def test_gradient_takeo_float32():
    dtype = np.float32
    t = takeo.copy()
    t.pixels = t.pixels.astype(dtype)
    grad_image = gradient(t)
    _check_assertions(grad_image, t.shape, t.n_channels * 2,
                      dtype)
    np_grad = _np_gradient(t.pixels)
    assert_allclose(grad_image.pixels, np_grad)


def test_gradient_double():
    dtype = np.float64
    p = example_image.astype(dtype)
    image = Image(p)
    grad_image = gradient(image)
    _check_assertions(grad_image, image.shape, image.n_channels * 2,
                      dtype)
    np_grad = np.gradient(p)
    assert_allclose(grad_image.pixels[0], np_grad[0])
    assert_allclose(grad_image.pixels[1], np_grad[1])


def test_gradient_takeo_double():
    t = takeo.copy()
    t.pixels = t.pixels.astype(np.float64)
    grad_image = gradient(t)

    np_grad = _np_gradient(t.pixels)
    assert_allclose(grad_image.pixels, np_grad)


@raises(TypeError)
def test_gradient_uint8_exception():
    image = Image(example_image.astype(np.uint8))
    gradient(image)


def _check_assertions(actual_image, expected_shape, expected_n_channels,
                      expected_type):
    assert (actual_image.pixels.dtype == expected_type)
    assert (type(actual_image) == Image)
    assert (actual_image.shape == expected_shape)
    assert (actual_image.n_channels == expected_n_channels)
