import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from menpo.transform import Rotation


def test_basic_2d_rotation():
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    rotation = Rotation(rotation_matrix)
    assert_allclose(np.array([0, -1]), rotation.apply(np.array([1, 0])))


def test_basic_2d_rotation_axis_angle():
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([0, 0, 1]))
    assert_allclose((90 * np.pi) / 180, angle)


def test_basic_3d_rotation():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0], [0, a, b], [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    starting_vector = np.array([0, 1, 0])
    transformed = rotation.apply(starting_vector)
    assert_allclose(np.array([0, a, -b]), transformed)


def test_basic_3d_rotation_axis_angle():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0], [0, a, b], [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    axis, angle = rotation.axis_and_angle_of_rotation()
    assert_allclose(axis, np.array([1, 0, 0]))
    assert_allclose((-30 * np.pi) / 180, angle)


def test_3d_rotation_inverse_eye():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0], [0, a, b], [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    transformed = rotation.compose_before(rotation.pseudoinverse())
    assert_allclose(np.eye(4), transformed.h_matrix, atol=1e-15)


def test_init_3d_from_quaternion():
    q = np.array([1.0, 0.0, 0.27, 0.0])
    r = Rotation.init_3d_from_quaternion(q)
    axis, angle = r.axis_and_angle_of_rotation()
    assert_allclose(r.axis_and_angle_of_rotation()[0], np.array([0.0, 1.0, 0.0]))
    assert np.round(angle * 180 / np.pi) == 30.0


def test_3d_rotation_as_vector():
    a = np.sqrt(3.0) / 2.0
    b = 0.5
    # this is a rotation of -30 degrees about the x axis
    rotation_matrix = np.array([[1, 0, 0], [0, a, b], [0, -b, a]])
    rotation = Rotation(rotation_matrix)
    assert_allclose(np.round(rotation.as_vector()[2:]), np.array([0.0, 0.0]))


def test_3d_rotation_n_parameters():
    assert Rotation.init_identity(3).n_parameters == 4


def test_2d_rotation_n_parameters_is_one_the_angle():
    assert Rotation.init_identity(2).n_parameters == 1
    assert Rotation.init_from_2d_ccw_angle(40).n_parameters == 1


def test_2d_rotation_as_vector_is_the_counter_clockwise_angle_in_radians():
    for degrees in [0, 30, 90, 135, -45, -170]:
        rotation = Rotation.init_from_2d_ccw_angle(degrees)
        assert_allclose(rotation.as_vector(), [np.deg2rad(degrees)], atol=1e-12)
    assert Rotation.init_identity(2).as_vector().shape == (1,)
    assert_allclose(Rotation.init_identity(2).as_vector(), [0.0])


def test_2d_rotation_as_vector_is_in_minus_pi_to_pi():
    # 270 degrees counter-clockwise is a quarter turn clockwise
    assert_allclose(
        Rotation.init_from_2d_ccw_angle(270).as_vector(), [-np.pi / 2], atol=1e-12
    )
    assert_allclose(
        Rotation.init_from_2d_ccw_angle(360 + 30).as_vector(),
        [np.deg2rad(30)],
        atol=1e-12,
    )


def test_2d_rotation_from_vector_builds_the_rotation_of_that_angle():
    for theta in [0.0, 0.3, -1.2, np.pi / 2, 2.9, -3.1]:
        rotation = Rotation.init_identity(2).from_vector(np.array([theta]))
        expected = Rotation.init_from_2d_ccw_angle(theta, degrees=False)
        assert_allclose(rotation.h_matrix, expected.h_matrix, atol=1e-15)


def test_2d_rotation_vector_round_trip():
    thetas = np.random.RandomState(0).uniform(-np.pi + 1e-3, np.pi - 1e-3, 50)
    for theta in thetas:
        rotation = Rotation.init_from_2d_ccw_angle(theta, degrees=False)
        again = Rotation.init_identity(2).from_vector(rotation.as_vector())
        assert_allclose(again.h_matrix, rotation.h_matrix, atol=1e-12)
        assert_allclose(again.as_vector(), [theta], atol=1e-12)


def test_2d_rotation_from_vector_does_not_change_the_rotation_it_is_called_on():
    rotation = Rotation.init_from_2d_ccw_angle(10)
    before = rotation.h_matrix.copy()
    other = rotation.from_vector(np.array([1.0]))
    assert_allclose(rotation.h_matrix, before)
    assert not np.allclose(other.h_matrix, before)


def test_2d_rotation_from_vector_inplace_sets_the_rotation():
    rotation = Rotation.init_identity(2)
    rotation._from_vector_inplace(np.array([np.pi / 2]))
    assert_allclose(rotation.apply(np.array([[1.0, 0.0]])), [[0.0, 1.0]], atol=1e-12)


def test_2d_rotation_from_vector_needs_exactly_one_parameter():
    rotation = Rotation.init_identity(2)
    for bad in [np.array([]), np.array([0.1, 0.2]), np.zeros((1, 1)), 0.5]:
        with raises(ValueError, match="1 angle parameter"):
            rotation.from_vector(bad)


def test_rotations_that_are_not_2d_or_3d_are_not_vectorizable():
    t = Rotation(np.eye(4))
    with raises(NotImplementedError, match="2D and 3D"):
        t.n_parameters
    with raises(NotImplementedError, match="2D and 3D"):
        t.as_vector()
    with raises(NotImplementedError, match="2D and 3D"):
        t.from_vector(np.zeros(1))


def test_3d_rotation_vector_is_unchanged():
    q = np.array([1.0, 0.0, 0.27, 0.0])
    rotation = Rotation.init_3d_from_quaternion(q)
    again = Rotation.init_identity(3).from_vector(rotation.as_vector())
    assert_allclose(again.h_matrix, rotation.h_matrix, atol=1e-12)
    with raises(ValueError, match="4 quaternion parameters"):
        Rotation.init_identity(3).from_vector(np.zeros(3))


def test_rotation2d_identity():
    assert_allclose(Rotation.init_identity(2).h_matrix, np.eye(3))


def test_rotation3d_identity():
    assert_allclose(Rotation.init_identity(3).h_matrix, np.eye(4))


def test_rotation3d_init_from_3d_ccw_angle_around_x():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_x(90).apply(np.array([0, 0, 1])),
        np.array([0, -1, 0]),
        atol=1e-6,
    )


def test_rotation3d_init_from_3d_ccw_angle_around_y():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_y(90).apply(np.array([0, 0, 1])),
        np.array([1, 0, 0]),
        atol=1e-6,
    )


def test_rotation3d_init_from_3d_ccw_angle_around_z():
    assert_allclose(
        Rotation.init_from_3d_ccw_angle_around_z(90).apply(np.array([0, 1, 0])),
        np.array([-1, 0, 0]),
        atol=1e-6,
    )
