import numpy as np
from numpy.testing import assert_allclose
from menpo.image import BooleanImage
from menpo.shape import PointCloud


def test_boolean_image_constrain_landmarks():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    mask.landmarks['test'] = PointCloud(
        np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    mask.constrain_to_landmarks('test')
    assert_allclose(mask.pixels[1:-1, 1:-1], True)
    assert mask.n_true() == 64


def test_boolean_image_constrain_pointcloud_pwa():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    pc = PointCloud(np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    mask.constrain_to_pointcloud(pc, point_in_pointcloud='pwa')
    assert_allclose(mask.pixels[:, 1:-1, 1:-1], True)
    assert mask.n_true() == 64


def test_boolean_image_constrain_pointcloud_convex_hull():
    mask = BooleanImage.init_blank((10, 10), fill=False)
    pc = PointCloud(np.array([[1, 1], [8, 1], [8, 8], [1, 8]]))
    mask.constrain_to_pointcloud(pc, point_in_pointcloud='convex_hull')
    assert_allclose(mask.pixels[:, 2:-1, 2:-1], True)
    # Point on the boundary are OUTSIDE
    assert mask.n_true() == 56
