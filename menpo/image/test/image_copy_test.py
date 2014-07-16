import numpy as np

from menpo.image import Image, BooleanImage, MaskedImage
from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_image_copy():
    pixels = np.ones([10, 10, 1])
    landmarks = PointCloud(np.ones([3, 2]), copy=False)
    im = Image(pixels, copy=False)
    im.landmarks['test'] = landmarks
    im_copy = im.copy()

    assert (not is_same_array(im.pixels, im_copy.pixels))
    assert (not is_same_array(im_copy.landmarks['test'].lms.points,
                              im.landmarks['test'].lms.points))


def test_booleanimage_copy():
    pixels = np.ones([10, 10], dtype=np.bool)
    landmarks = PointCloud(np.ones([3, 2]), copy=False)
    im = BooleanImage(pixels, copy=False)
    im.landmarks['test'] = landmarks
    im_copy = im.copy()

    assert (not is_same_array(im.pixels, im_copy.pixels))
    assert (not is_same_array(im_copy.landmarks['test'].lms.points,
                              im.landmarks['test'].lms.points))


def test_maskedimage_copy():
    pixels = np.ones([10, 10, 1])
    landmarks = PointCloud(np.ones([3, 2]), copy=False)
    im = MaskedImage(pixels, copy=False)
    im.landmarks['test'] = landmarks
    im_copy = im.copy()

    assert (not is_same_array(im.pixels, im_copy.pixels))
    assert (not is_same_array(im_copy.landmarks['test'].lms.points,
                              im.landmarks['test'].lms.points))
