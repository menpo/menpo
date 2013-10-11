import numpy as np
from pybug.image import *
from nose.tools import raises
from numpy.testing import assert_equal


def test_blank_depthimage():
    mask = np.zeros((10, 10))
    im = DepthImage.blank((10, 10), mask=mask)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 1
    assert np.all(im.mask.pixels == 0.0)

    im = DepthImage.blank((10, 10), fill=2.0)
    assert np.all(im.pixels == 2.0)


@raises(ValueError)
def test_blank_depthimage_bad_channels():
    mask = np.ones((10, 10))
    DepthImage.blank((10, 10), mask=mask, n_channels=2)


def test_blank_shapeimage():
    mask = np.zeros((10, 10))
    im = ShapeImage.blank((10, 10), mask=mask, n_channels=3)
    # the z values should be zero
    assert np.all(im.pixels[..., 2] == 0.0)
    # the x,y values should be indices, spot check this
    assert_equal(im.pixels[2, 6, :2], np.array([2, 6]))
    assert im.n_channels == 3
    assert np.all(im.mask.pixels == 0.0)

    im = ShapeImage.blank((10, 10), fill=2.0, n_channels=3)
    assert np.all(im.pixels[..., 2] == 2.0)


@raises(ValueError)
def test_blank_shapeimage_bad_channels():
    mask = np.ones((10, 10))
    DepthImage.blank((10, 10), mask=mask, n_channels=2)


def test_blank_intensityimage():
    mask = np.zeros((10, 10))
    im = IntensityImage.blank((10, 10), mask=mask)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 1
    assert np.all(im.mask.pixels == 0.0)

    im = IntensityImage.blank((10, 10), fill=0.5)
    assert np.all(im.pixels == 0.5)


@raises(ValueError)
def test_blank_intensityimage_bad_channels():
    mask = np.ones((10, 10))
    IntensityImage.blank((10, 10), mask=mask, n_channels=2)


def test_blank_rgbimage():
    mask = np.zeros((10, 10))
    im = RGBImage.blank((10, 10), mask=mask, n_channels=3)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 3
    assert np.all(im.mask.pixels == 0.0)

    im = RGBImage.blank((10, 10), fill=0.5, n_channels=3)
    assert np.all(im.pixels == 0.5)


@raises(ValueError)
def test_blank_rgbimage_bad_channels():
    mask = np.ones((10, 10))
    RGBImage.blank((10, 10), mask=mask, n_channels=2)


def test_blank_maskedndimage():
    mask = np.zeros((10, 10))
    im = MaskedNDImage.blank((10, 10), mask=mask, n_channels=10)
    assert np.all(im.pixels == 0.0)
    assert im.n_channels == 10
    assert np.all(im.mask.pixels == 0.0)

    im = MaskedNDImage.blank((10, 10), fill=2.0, n_channels=10)
    assert np.all(im.pixels == 2.0)