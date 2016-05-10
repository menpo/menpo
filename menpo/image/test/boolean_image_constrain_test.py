import numpy as np
from numpy.testing import assert_allclose
from menpo.image import BooleanImage
from menpo.shape import PointCloud


def test_boolean_image_constrain_landmarks():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    mask.landmarks['test'] = PointCloud(
        np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    new_mask = mask.constrain_to_landmarks('test')
    assert_allclose(new_mask.pixels[1:-1, 1:-1], True)
    assert new_mask.n_true() == 64


def test_boolean_image_constrain_pointcloud_pwa():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    pc = PointCloud(np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    new_mask = mask.constrain_to_pointcloud(pc, point_in_pointcloud='pwa')
    assert_allclose(new_mask.pixels[:, 1:-1, 1:-1], True)
    assert new_mask.n_true() == 64


def test_boolean_image_constrain_pointcloud_convex_hull():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    pc = PointCloud(np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    new_mask = mask.constrain_to_pointcloud(pc,
                                            point_in_pointcloud='convex_hull')
    assert_allclose(new_mask.pixels[:, 2:-1, 2:-1], True)
    # Points on the boundary are OUTSIDE
    assert new_mask.n_true() == 56


def test_boolean_image_init_from_pointcloud():
    pc = PointCloud(np.array([[5, 5], [5, 20], [20, 20]]))
    im = BooleanImage.init_from_pointcloud(pc, fill=False, constrain=False)
    assert im.n_true() == 0
    assert im.shape == (15, 15)


def test_boolean_image_init_from_pointcloud_constraint():
    pc = PointCloud(np.array([[5, 5], [5, 20], [20, 20]]))
    im = BooleanImage.init_from_pointcloud(pc, fill=False, constrain=True)
    assert im.n_true() == 120
    assert im.shape == (15, 15)


def test_boolean_image_init_from_pointcloud_constrain_all_true():
    pc = PointCloud(np.array([[5, 5], [5, 20], [20, 20]]))
    im = BooleanImage.init_from_pointcloud(pc, fill=True, constrain=True)
    assert im.n_true() == 120
    assert im.shape == (15, 15)
