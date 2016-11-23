import numpy as np
from menpo.shape import PointCloud
from menpo.transform import WithDims


def test_withdims_transform():
    pc_3d = PointCloud(np.random.random((14, 3)))
    pc_2d = WithDims([0, 1]).apply(pc_3d)
    assert np.all(pc_3d.points[:, :2] == pc_2d.points)


def test_withdims_single_axis_transform():
    pc_3d = PointCloud(np.random.random((14, 3)))
    pc_1d = WithDims(0).apply(pc_3d)
    assert np.all(pc_3d.points[:, :0] == pc_1d.points)
    assert pc_1d.n_dims == 1
    assert pc_1d.n_points == pc_3d.n_points


def test_with_dims_method():
    pc_3d = PointCloud(np.random.random((14, 3)))
    pc_2d = pc_3d.with_dims([0, 1])
    assert np.all(pc_3d.points[:, :2] == pc_2d.points)


def test_with_dims_flip():
    pc_3d = PointCloud(np.random.random((14, 3)))
    pc_2d = pc_3d.with_dims([1, 0])
    assert np.all(pc_3d.points[:, :2] == pc_2d.points[:, ::-1])


def test_with_dims_repeat():
    pc_3d = PointCloud(np.random.random((14, 3)))
    pc_3d_rep = pc_3d.with_dims([0, 0, 1])
    assert np.all(pc_3d.points[:, 0] == pc_3d_rep.points[:, 0])
    assert np.all(pc_3d.points[:, 0] == pc_3d_rep.points[:, 1])
    assert np.all(pc_3d.points[:, 1] == pc_3d_rep.points[:, 2])
