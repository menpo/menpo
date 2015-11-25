import warnings
import numpy as np
from nose.tools import raises
from numpy.testing import assert_allclose
from menpo.shape import PointCloud, bounding_box
from menpo.testing import is_same_array


def test_pointcloud_creation():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    PointCloud(points)


def test_pointcloud_init_2d_grid():
    pc = PointCloud.init_2d_grid([10, 10])
    assert pc.n_points == 100
    assert pc.n_dims == 2
    assert_allclose(pc.range(), [9, 9])


def test_pointcloud_init_2d_grid_single_spacing():
    pc = PointCloud.init_2d_grid([10, 10], spacing=2)
    assert pc.n_points == 100
    assert pc.n_dims == 2
    assert_allclose(pc.range(), [18, 18])


def test_pointcloud_init_2d_grid_unequal_spacing():
    pc = PointCloud.init_2d_grid([10, 10], spacing=(2., 3))
    assert pc.n_points == 100
    assert pc.n_dims == 2
    assert_allclose(pc.range(), [18, 27])


@raises(ValueError)
def test_pointcloud_init_2d_grid_3d_raises():
    PointCloud.init_2d_grid([10, 10, 10])


@raises(ValueError)
def test_pointcloud_init_2d_grid_3d_spacing_raises():
    PointCloud.init_2d_grid([10, 10], spacing=[1, 1, 1])


@raises(ValueError)
def test_pointcloud_init_2d_grid_incorrect_type_spacing_raises():
    PointCloud.init_2d_grid([10, 10], spacing={})


def test_pointcloud_has_nan_values():
    pcloud = PointCloud(np.random.rand(3, 2), copy=False)
    pcloud.points[0, 0] = np.nan
    assert pcloud.has_nan_values()


def test_pointcloud_no_nan_values():
    pcloud = PointCloud(np.random.rand(3, 2), copy=False)
    assert not pcloud.has_nan_values()


def test_pointcloud_copy_method():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]])
    landmarks = PointCloud(np.ones([3, 3]), copy=False)

    p = PointCloud(points, copy=False)
    p.landmarks['test'] = landmarks
    p_copy = p.copy()

    assert (not is_same_array(p_copy.points, p.points))
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


def test_pointcloud_copy_warning():
    points = np.array([[1, 2, 3],
                       [1, 1, 1]], order='F')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        PointCloud(points, copy=False)
        assert len(w) == 1


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


def test_pointcloud_centre():
    points = np.array([[0, 0],
                       [1., 1],
                       [0, 2]])
    pc = PointCloud(points)
    c = pc.centre()
    assert_allclose(c, [1. / 3., 1.])


def test_pointcloud_centre_of_bounds():
    points = np.array([[0, 0],
                       [1., 1],
                       [0, 2]])
    pc = PointCloud(points)
    cb = pc.centre_of_bounds()
    assert_allclose(cb, [0.5, 1.])
    assert 1


def test_pointcloud_bounding_box():
    points = np.array([[0, 0],
                       [1., 1],
                       [0, 2]])
    pc = PointCloud(points)
    bb = pc.bounding_box()
    bb_bounds = bb.bounds()
    assert_allclose(bb_bounds[0], [0., 0.])
    assert_allclose(bb_bounds[1], [1., 2.])


@raises(ValueError)
def test_pointcloud_bounding_box_3d_fail():
    points = np.array([[0, 0, 0],
                       [1, 1, 1]])
    pc = PointCloud(points)
    pc.bounding_box()


def test_bounding_box_creation():
    bb = bounding_box([0, 0], [1, 1])
    assert_allclose(bb.points, [[0, 0], [1, 0], [1, 1], [0, 1]])
