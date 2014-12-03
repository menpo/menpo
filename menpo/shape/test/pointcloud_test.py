import warnings
import numpy as np
from menpo.shape import PointCloud, BoundingBox
from menpo.testing import is_same_array
from nose.tools import raises


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


def test_bounding_box_creation():
    points = np.array([[0, 0],
                       [1, 1]])
    BoundingBox(points)


@raises(ValueError)
def test_bounding_box_n_points_raises_value_error():
    points = np.array([[0, 0],
                       [1, 1],
                       [0, 1],
                       [1, 0]])
    BoundingBox(points)


def test_bounding_box_min():
    points = np.array([[0, 0],
                       [1, 1]])
    bb = BoundingBox(points)
    assert np.all(bb.min == np.array([0, 0]))


def test_bounding_box_max():
    points = np.array([[0, 0],
                       [1, 1]])
    bb = BoundingBox(points)
    assert np.all(bb.max == np.array([1, 1]))


def test_bounding_box_box():
    points = np.array([[0, 0],
                       [1, 1]])
    bb = BoundingBox(points)
    box = bb.box()
    assert np.all(box.points == np.array([[0, 0],
                                          [1, 1],
                                          [0, 1],
                                          [1, 0]]))


# creating a k-dims bounding box is fine
def test_bounding_box_3_dims():
    points = np.array([[0, 0, 0],
                       [1, 1, 1]])
    BoundingBox(points)


# creating a k-dims .box() is not supported
@raises(ValueError)
def test_bounding_box_3_dims_box_raises_value_error():
    points = np.array([[0, 0, 0],
                       [1, 1, 1]])
    BoundingBox(points).box()


def test_bounding_box_str():
    points = np.array([[0, 0],
                       [1, 1]])
    assert (str(BoundingBox(points)) ==
            'BoundingBox: min: [0 0], max: [1 1], n_dims: 2')
