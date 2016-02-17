import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

from menpo.transform import Rotation


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
    assert_allclose((90 * np.pi) / 180, angle)


def test_basic_3d_rotation():
    a = np.sqrt(3.0) / 2.0
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
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([1, 0, 0]))
    assert_allclose((-30 * np.pi) / 180, angle)


def test_3d_rotation_inverse_eye():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    transformed = rotation.compose_before(rotation.pseudoinverse())
    assert_allclose(np.eye(4), transformed.h_matrix, atol=1e-15)


@raises(NotImplementedError)
def test_rotation3d_from_vector_raises_notimplementederror():
    Rotation.init_identity(3).from_vector(0)


@raises(NotImplementedError)
def test_rotation3d_as_vector_raises_notimplementederror():
    Rotation.init_identity(3).as_vector()


@raises(NotImplementedError)
def test_rotation2d_n_parameters_raises_notimplementederror():
    rot_matrix = np.eye(2)
    t = Rotation(rot_matrix)
    t.n_parameters


@raises(NotImplementedError)
def test_rotation3d_n_parameters_raises_notimplementederror():
    rot_matrix = np.eye(3)
    t = Rotation(rot_matrix)
    # Throws exception
    t.n_parameters


def test_rotation2d_identity():
    assert_allclose(Rotation.init_identity(2).h_matrix, np.eye(3))


def test_rotation3d_identity():
    assert_allclose(Rotation.init_identity(3).h_matrix, np.eye(4))
