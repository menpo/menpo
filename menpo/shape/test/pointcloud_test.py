from nose.tools import raises
import numpy as np

from menpo.shape import PointCloud
from menpo.testing import is_same_array


def test_pointcloud_creation():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    PointCloud(points)


def test_pointcloud_copy_method():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    landmarks = PointCloud(np.ones([3, 3]), copy=False)

    p = PointCloud(points, copy=False)
    p.landmarks['test'] = landmarks
    p_copy = p.copy()

    assert (not is_same_array(p_copy.points, p.points))
    assert (p_copy.landmarks['test']._target is p_copy)
    assert (not is_same_array(p_copy.landmarks['test'].lms.points,
                              p.landmarks['test'].lms.points))


def test_pointcloud_copy_false():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    p = PointCloud(points, copy=False)
    assert (is_same_array(p.points, points))


def test_pointcloud_copy_true():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    p = PointCloud(points)
    assert (not is_same_array(p.points, points))


@raises(Warning)
def test_pointcloud_copy_warning():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]], order='F')
    PointCloud(points, copy=False)


def test_pointcloud_n_dims():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    assert (pc.n_dims == 3)


def test_pointcloud_n_points():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    assert (pc.n_points == 2)


def test_pointcloud_flatten_rebuild():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    flattened = pc.as_vector()
    new_pc = pc.from_vector(flattened)
    assert (np.all(new_pc.n_dims == pc.n_dims))
    assert (np.all(new_pc.n_points == pc.n_points))
    assert (np.all(pc.points == new_pc.points))
