import numpy as np
from numpy.testing import assert_allclose
from pybug.transform.affine import Rotation, Translation
from pybug.exceptions import DimensionalityError
from nose.tools import raises


@raises(DimensionalityError)
def test_1d():
    t_vec = np.array([1])
    Translation(t_vec)


@raises(DimensionalityError)
def test_5d():
    t_vec = np.ones(5)
    Translation(t_vec)


def test_translation():
    t_vec = np.array([1, 2, 3])
    starting_vector = np.random.rand(10, 3)
    transform = Translation(t_vec)
    transformed = transform.apply(starting_vector)
    assert_allclose(starting_vector + t_vec, transformed)


def test_basic_2d_rotation():
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    rotation = Rotation(rotation_matrix)
    assert_allclose(np.array([0, -1]), rotation.apply(np.array([1, 0])))


def test_basic_2d_rotation_axis_angle():
    rotation_matrix = np.array([[0, 1],
                                [-1, 0]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([0, 0, 1]))
    assert_allclose((90 * np.pi)/180, angle)


def test_basic_3d_rotation():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    starting_vector = np.array([0, 1, 0])
    transformed = rotation.apply(starting_vector)
    assert_allclose(np.array([0, a, -b]), transformed)


def test_basic_3d_rotation_axis_angle():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([1, 0, 0]))
    assert_allclose((-30 * np.pi)/180, angle)


def test_3d_rotation_inverse_eye():
    a = np.sqrt(3.0)/2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    transformed = rotation.compose(rotation.inverse)
    print transformed.homogeneous_matrix
    assert_allclose(np.eye(4), transformed.homogeneous_matrix, atol=1e-15)
