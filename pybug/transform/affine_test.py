import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pybug.transform.affine import Rotation, Translation, \
    AffineTransform, SimilarityTransform, NonUniformScale, \
    Rotation2D, Rotation3D, UniformScale2D, UniformScale3D, UniformScale, Scale
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

jac_solution2d = np.array(
    [[[0.,  0.],
    [0.,  0.],
    [0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.]],
    [[0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.]],
    [[0.,  0.],
    [0.,  0.],
    [2.,  0.],
    [0.,  2.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [0.,  0.],
    [0.,  0.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.],
    [1.,  0.],
    [0.,  1.]],
    [[1.,  0.],
    [0.,  1.],
    [2.,  0.],
    [0.,  2.],
    [1.,  0.],
    [0.,  1.]]])

jac_solution3d = np.array(
    [[[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [0.,  0.,  0.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]],
    [[1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [2.,  0.,  0.],
    [0.,  2.,  0.],
    [0.,  0.,  2.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.],
    [1.,  0.,  0.],
    [0.,  1.,  0.],
    [0.,  0.,  1.]]])


def test_affine_jacobian_2d_with_positions():
    params = np.array([0, 0.1, 0.2, 0, 30, 70])
    t = AffineTransform.from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2]])
    dW_dp = t.jacobian(explicit_pixel_locations)
    assert_equal(dW_dp, jac_solution2d)


def test_affine_jacobian_3d_with_positions():
    params = np.ones(12)
    t = AffineTransform.from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [0, 2, 0],
        [0, 2, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
        [1, 2, 0],
        [1, 2, 1]])
    dW_dp = t.jacobian(explicit_pixel_locations)
    assert_equal(dW_dp, jac_solution3d)


def test_similarity_2d_from_vector():
    params = np.array([0.2, 0.1, 1, 2])
    homo = np.array([[params[0] + 1, -params[1], params[2]],
                     [params[1], params[0] + 1, params[3]],
                     [0, 0, 1]])

    sim = SimilarityTransform.from_vector(params)

    assert_equal(sim.homogeneous_matrix, homo)


def test_similarity_2d_as_vector():
    params = np.array([0.2, 0.1, 1.0, 2.0])
    homo = np.array([[params[0] + 1.0, -params[1], params[2]],
                     [params[1], params[0] + 1.0, params[3]],
                     [0.0, 0.0, 1.0]])

    vec = SimilarityTransform(homo).as_vector()

    assert_allclose(vec, params)


def test_translation_2d_from_vector():
    params = np.array([1, 2])
    homo = np.array([[1, 0, params[0]],
                     [0, 1, params[1]],
                     [0, 0, 1]])

    tr = Translation.from_vector(params)

    assert_equal(tr.homogeneous_matrix, homo)


def test_translation_2d_as_vector():
    params = np.array([1, 2])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_translation_3d_from_vector():
    params = np.array([1, 2, 3])
    homo = np.array([[1, 0, 0, params[0]],
                     [0, 1, 0, params[1]],
                     [0, 0, 1, params[2]],
                     [0, 0, 0, 1]])

    tr = Translation.from_vector(params)

    assert_equal(tr.homogeneous_matrix, homo)


def test_translation_3d_as_vector():
    params = np.array([1, 2, 3])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_uniformscale2d_from_vector():
    scale = 2
    homo = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]])

    tr = UniformScale2D.from_vector(scale)

    assert_equal(tr.homogeneous_matrix, homo)


def test_uniformscale2d_as_vector():
    scale = 2
    vec = UniformScale2D(scale).as_vector()
    assert_allclose(vec, scale)


def test_nonuniformscale2d_from_vector():
    scale = np.array([1, 2])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])

    tr = NonUniformScale.from_vector(scale)

    assert_equal(tr.homogeneous_matrix, homo)


def test_nonuniformscale2d_as_vector():
    scale = np.array([1, 2])
    vec = NonUniformScale(scale).as_vector()
    assert_allclose(vec, scale)


def test_uniformscale3d_from_vector():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    tr = UniformScale3D.from_vector(scale)

    assert_equal(tr.homogeneous_matrix, homo)


def test_uniformscale3d_as_vector():
    scale = 2

    vec = UniformScale3D(scale).as_vector()

    assert_allclose(vec, scale)


def test_uniformscale_build_2d():
    scale = 2
    homo = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]])

    tr = UniformScale(scale, 2)

    assert(isinstance(tr, UniformScale2D))
    assert_equal(tr.homogeneous_matrix, homo)


def test_uniformscale_build_3d():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    tr = UniformScale(scale, 3)

    assert(isinstance(tr, UniformScale3D))
    assert_equal(tr.homogeneous_matrix, homo)

@raises(DimensionalityError)
def test_uniformscale_build_4d_raise_dimensionalityerror():
    UniformScale(1, 4)


def test_scale_build_2d_uniform_pass_dim():
    scale = 2
    ndim = 2
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale2D))


def test_scale_build_3d_uniform_pass_dim():
    scale = 2
    ndim = 3
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale3D))


def test_scale_build_2d_nonuniform():
    scale = np.array([1, 2])
    tr = Scale(scale)

    assert(isinstance(tr, NonUniformScale))


def test_scale_build_2d_uniform_from_vec():
    scale = np.array([2, 2])
    tr = Scale(scale)

    assert(isinstance(tr, UniformScale2D))


@raises(ValueError)
def test_scale_zero_scale_raise_valuerror():
    Scale(np.array([1, 0]))


def test_rotation2d_from_vector():
    theta = np.pi / 2
    homo = np.array([[0.0, -1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

    tr = Rotation2D.from_vector(theta)

    assert_allclose(tr.homogeneous_matrix, homo, atol=1**-15)


def test_rotation2d_as_vector():
    theta = np.pi / 2
    rot_matrix = np.array([[0.0, -1.0],
                          [1.0, 0.0]])

    vec = Rotation2D(rot_matrix).as_vector()

    assert_allclose(vec, theta)


@raises(NotImplementedError)
def test_rotation3d_from_vector_raises_notimplementederror():
    Rotation3D.from_vector(0)


@raises(NotImplementedError)
def test_rotation3d_as_vector_raises_notimplementederror():
    homo = np.eye(3)
    Rotation3D(homo).as_vector()


def test_affine_2d_n_parameters():
    homo = np.eye(3)
    t = AffineTransform(homo)
    assert(t.n_parameters == 6)


def test_affine_3d_n_parameters():
    homo = np.eye(4)
    t = AffineTransform(homo)
    assert(t.n_parameters == 12)


def test_similarity_2d_n_parameters():
    homo = np.eye(3)
    t = SimilarityTransform(homo)
    assert(t.n_parameters == 4)


@raises(NotImplementedError)
def test_similarity_3d_n_parameters_raises_notimplementederror():
    homo = np.eye(4)
    t = SimilarityTransform(homo)
    # Raises exception
    t.n_parameters


def test_uniformscale2d_n_parameters():
    scale = 2
    t = UniformScale2D(scale)
    assert(t.n_parameters == 1)


def test_uniformscale3d_n_parameters():
    scale = 2
    t = UniformScale3D(scale)
    assert(t.n_parameters == 1)


def test_nonuniformscale_2d_n_parameters():
    scale = np.array([1, 2])
    t = NonUniformScale(scale)
    assert(t.n_parameters == 2)


def test_translation_2d_n_parameters():
    trans = np.array([1, 2])
    t = Translation(trans)
    assert(t.n_parameters == 2)


def test_translation_3d_n_parameters():
    trans = np.array([1, 2, 3])
    t = Translation(trans)
    assert(t.n_parameters == 3)


def test_rotation2d_n_parameters():
    rot_matrix = np.eye(2)
    t = Rotation2D(rot_matrix)
    assert(t.n_parameters == 1)

@raises(NotImplementedError)
def test_rotation3d_n_parameters_raises_notimplementederror():
    rot_matrix = np.eye(3)
    t = Rotation3D(rot_matrix)
    # Throws exception
    t.n_parameters