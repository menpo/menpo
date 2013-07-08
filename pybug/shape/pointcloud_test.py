import numpy as np
from pybug.shape import PointCloud


def test_pointcloud_creation():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    PointCloud(points)


def test_pointcloud_n_dims():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    assert(pc.n_dims == 3)


def test_pointcloud_n_points():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    assert(pc.n_points == 2)


def test_pointcloud_flatten_rebuild():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pc = PointCloud(points)
    flattened = pc.as_flattened()
    new_pc = PointCloud.from_flattened_with_instance(flattened, pc)
    assert(np.all(new_pc.n_dims == pc.n_dims))
    assert(np.all(new_pc.n_points == pc.n_points))
    assert(np.all(pc.points == new_pc.points))
