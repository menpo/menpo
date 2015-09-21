import numpy as np
from numpy.testing import assert_allclose
from menpo.shape import PointCloud, mean_pointcloud, TriMesh


def test_mean_pointcloud():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    pcs = [PointCloud(points), PointCloud(points + 2)]
    mean_pc = mean_pointcloud(pcs)
    assert isinstance(mean_pc, PointCloud)
    assert_allclose(mean_pc.points, points + 1)


def test_mean_pointcloud_type():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    trilist = np.array([0, 1, 2])
    pcs = [TriMesh(points, trilist), TriMesh(points + 2, trilist)]
    mean_pc = mean_pointcloud(pcs)
    assert isinstance(mean_pc, TriMesh)
    assert_allclose(mean_pc.points, points + 1)
