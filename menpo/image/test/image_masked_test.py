import numpy as np
from nose.tools import raises
from numpy.testing import assert_allclose

from menpo.shape import PointCloud
from menpo.image import MaskedImage, BooleanImage


def test_constrain_mask_to_landmarks_pwa():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0.0, 0.0], [5.0, 0.0],
                                                [5.0, 5.0], [0.0, 5.0]]))
    img.constrain_mask_to_landmarks(group='box')

    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(img.mask.n_true() == 36)
    assert_allclose(img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_pwa_batched():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0.0, 0.0], [5.0, 0.0],
                                                [5.0, 5.0], [0.0, 5.0]]))
    img.constrain_mask_to_landmarks(group='box', batch_size=2)

    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(img.mask.n_true() == 36)
    assert_allclose(img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_convex_hull():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    img.constrain_mask_to_landmarks(group='box',
                                    point_in_pointcloud='convex_hull')
    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, 1:6] = True
    assert(img.mask.n_true() == 30)
    assert_allclose(img.mask.pixels, example_mask.pixels)


def test_constrain_mask_to_landmarks_callable():
    def bounding_box(_, indices):
        return np.ones(indices.shape[0], dtype=np.bool)

    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0.], [5., 0.],
                                                [5., 5.], [0., 5.]]))
    img.constrain_mask_to_landmarks(group='box',
                                    point_in_pointcloud=bounding_box)
    example_mask = BooleanImage.init_blank((10, 10), fill=False)
    example_mask.pixels[0, :6, :6] = True
    assert(img.mask.n_true() == 36)
    assert_allclose(img.mask.pixels, example_mask.pixels)


@raises(ValueError)
def test_constrain_mask_to_landmarks_non_2d():
    img = MaskedImage.init_blank((10, 10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0., 0.]]))
    img.constrain_mask_to_landmarks()


@raises(ValueError)
def test_constrain_mask_to_landmarks_unknown_key():
    img = MaskedImage.init_blank((10, 10))
    img.landmarks['box'] = PointCloud(np.array([[0., 0., 0.]]))
    img.constrain_mask_to_landmarks(point_in_pointcloud='unknown')


def test_erode():
    img = MaskedImage.init_blank((10, 10))
    img2 = img.erode()
    assert(img2.mask.n_true() == 64)
    img3 = img.erode(n_pixels=3)
    assert(img3.mask.n_true() == 16)


def test_dilate():
    img = MaskedImage.init_blank((10, 10))
    img = img.erode(n_pixels=3)
    img2 = img.dilate()
    assert(img2.mask.n_true() == 32)
    img3 = img.dilate(n_pixels=3)
    assert(img3.mask.n_true() == 76)


def test_init_from_rolled_channels():
    p = np.empty([50, 60, 3])
    im = MaskedImage.init_from_rolled_channels(p)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60


def test_init_from_rolled_channels_masked():
    p = np.empty([50, 60, 3])
    example_mask = BooleanImage.init_blank((50, 60), fill=False)
    example_mask.pixels[0, :6, :6] = True

    im = MaskedImage.init_from_rolled_channels(p, mask=example_mask)
    assert im.n_channels == 3
    assert im.height == 50
    assert im.width == 60
    assert im.mask.n_true() == 36
