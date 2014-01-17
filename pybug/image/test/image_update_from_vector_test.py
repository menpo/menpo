import numpy as np
from numpy.testing import assert_allclose
from pybug.image import *
from nose.tools import raises


def update_im_from_vector(im):
    new_values = np.random.random(im.pixels.shape)
    im.from_vector_inplace(new_values.flatten())
    assert im.pixels.shape == new_values.shape
    return new_values


def test_depthimage_update_from_vector():
    im = DepthImage.blank((10, 10))
    # Force the lazy construction of the old mesh
    im.mesh
    new_values = update_im_from_vector(im)
    assert_allclose(im.mesh.points[:, 2], new_values.flatten())


def test_shapeimage_update_from_vector():
    old_values = np.random.random((10, 10, 3))
    im = ShapeImage(old_values)
    # Force the lazy construction of the old mesh
    im.mesh
    new_values = update_im_from_vector(im)
    assert_allclose(im.mesh.points.flatten(), new_values.flatten())


def test_1channel_update_from_vector():
    im = MaskedImage.blank((10, 10))
    update_im_from_vector(im)


def test_3channel_update_from_vector():
    im = MaskedImage.blank((10, 10), n_channels=3)
    update_im_from_vector(im)


def test_maskedndimage_update_from_vector():
    im = MaskedImage.blank((10, 10), n_channels=10)
    update_im_from_vector(im)