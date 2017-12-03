import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

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


def test_init_3d_from_quaternion():
    q = np.array([1., 0., 0.27, 0.])
    r = Rotation.init_3d_from_quaternion(q)
    axis, angle = r.axis_and_angle_of_rotation()
    assert_allclose(r.axis_and_angle_of_rotation()[0], np.array([0., 1., 0.]))
    assert np.round(angle * 180 / np.pi) == 30.


def test_3d_rotation_as_vector():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0],
                                [0, a, b],
                                [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    assert_allclose(np.round(rotation.as_vector()[2:]), np.array([0., 0.]))


def test_3d_rotation_n_parameters():
    assert Rotation.init_identity(3).n_parameters == 4


def test_rotation2d_from_vector_raises_notimplementederror():
    with raises(NotImplementedError):
        Rotation.init_identity(2).from_vector(0)


def test_rotation2d_as_vector_raises_notimplementederror():
    with raises(NotImplementedError):
        Rotation.init_identity(2).as_vector()


def test_rotation2d_n_parameters_raises_notimplementederror():
    rot_matrix = np.eye(2)
    t = Rotation(rot_matrix)
    with raises(NotImplementedError):
        t.n_parameters


def test_rotation2d_identity():
    assert_allclose(Rotation.init_identity(2).h_matrix, np.eye(3))


def test_rotation3d_identity():
    assert_allclose(Rotation.init_identity(3).h_matrix, np.eye(4))


def test_rotation3d_init_from_3d_ccw_angle_around_x():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_x(90).apply(np.array([0, 0, 1])),
        np.array([0, -1, 0]), atol=1e-6)


def test_rotation3d_init_from_3d_ccw_angle_around_y():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_y(90).apply(np.array([0, 0, 1])),
        np.array([1, 0, 0]), atol=1e-6)


def test_rotation3d_init_from_3d_ccw_angle_around_z():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_z(90).apply(np.array([0, 1, 0])),
        np.array([-1, 0, 0]), atol=1e-6)
