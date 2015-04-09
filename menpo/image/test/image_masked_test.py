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
