import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from menpo.transform import (Affine, Similarity, Rotation, Scale,
                             NonUniformScale, UniformScale, Translation,
                             Homogeneous)
from nose.tools import raises


@raises(ValueError)
def test_1d_translation():
    t_vec = np.array([1])
    Translation(t_vec)


@raises(ValueError)
def test_5d_translation():
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
    transformed = rotation.compose_before(rotation.pseudoinverse())
    assert_allclose(np.eye(4), transformed.h_matrix, atol=1e-15)


def test_basic_2d_affine():
    linear_component = np.array([[1, -6],
                                 [-3, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    affine = Affine(h_matrix)
    x = np.array([[0, 1],
                  [1, 1],
                  [-1, -5],
                  [3, -5]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = affine.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = affine.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_basic_3d_affine():
    linear_component = np.array([[1, 6, -4],
                                 [-3, -2, 5],
                                 [5, -1, 3]])
    translation_component = np.array([7, -8, 9])
    h_matrix = np.eye(4, 4)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    affine = Affine(h_matrix)
    x = np.array([[0, 1,  2],
                  [1, 1, 1],
                  [-1, 2, -5],
                  [1, -5, -1]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = affine.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = affine.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_basic_2d_similarity():
    linear_component = np.array([[2, -6],
                                 [6, 2]])
    translation_component = np.array([7, -8])
    h_matrix = np.eye(3, 3)
    h_matrix[:-1, :-1] = linear_component
    h_matrix[:-1, -1] = translation_component
    similarity = Similarity(h_matrix)
    x = np.array([[0, 1],
                  [1, 1],
                  [-1, -5],
                  [3, -5]])
    # transform x explicitly
    solution = np.dot(x, linear_component.T) + translation_component
    # transform x using the affine transform
    result = similarity.apply(x)
    # check that both answers are equivalent
    assert_allclose(solution, result)
    # create several copies of x
    x_copies = np.array([x, x, x, x, x, x, x, x])
    # transform all of copies at once using the affine transform
    results = similarity.apply(x_copies)
    # check that all copies have been transformed correctly
    for r in results:
        assert_allclose(solution, r)


def test_similarity_2d_from_vector():
    params = np.array([0.2, 0.1, 1, 2])
    homo = np.array([[params[0] + 1, -params[1], params[2]],
                     [params[1], params[0] + 1, params[3]],
                     [0, 0, 1]])

    sim = Similarity.init_identity(2).from_vector(params)

    assert_equal(sim.h_matrix, homo)


def test_similarity_2d_as_vector():
    params = np.array([0.2, 0.1, 1.0, 2.0])
    homo = np.array([[params[0] + 1.0, -params[1], params[2]],
                     [params[1], params[0] + 1.0, params[3]],
                     [0.0, 0.0, 1.0]])

    vec = Similarity(homo).as_vector()

    assert_allclose(vec, params)


def test_translation_2d_from_vector():
    params = np.array([1, 2])
    homo = np.array([[1, 0, params[0]],
                     [0, 1, params[1]],
                     [0, 0, 1]])

    tr = Translation.init_identity(2).from_vector(params)

    assert_equal(tr.h_matrix, homo)


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

    tr = Translation.init_identity(3).from_vector(params)

    assert_equal(tr.h_matrix, homo)


def test_translation_3d_as_vector():
    params = np.array([1, 2, 3])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_uniformscale2d_update_from_vector():
    # make a uniform scale of 1, 2 dimensional
    uniform_scale = UniformScale(1, 2)
    new_scale = 2
    homo = np.array([[new_scale, 0, 0],
                     [0, new_scale, 0],
                     [0, 0, 1]])

    uniform_scale._from_vector_inplace(new_scale)
    assert_equal(uniform_scale.h_matrix, homo)


def test_uniformscale2d_as_vector():
    scale = 2
    vec = UniformScale(scale, 2).as_vector()
    assert_allclose(vec, scale)


def test_nonuniformscale2d_from_vector():
    scale = np.array([1, 2])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])

    tr = NonUniformScale.init_identity(2).from_vector(scale)

    assert_equal(tr.h_matrix, homo)


def test_nonuniformscale2d_update_from_vector():
    scale = np.array([3, 4])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])
    tr = NonUniformScale(np.array([1, 2]))
    tr._from_vector_inplace(scale)
    assert_equal(tr.h_matrix, homo)


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

    uniform_scale = UniformScale(1, 3)
    tr = uniform_scale.from_vector(scale)
    assert_equal(tr.h_matrix, homo)


def test_uniformscale3d_as_vector():
    scale = 2
    vec = UniformScale(scale, 3).as_vector()
    assert_allclose(vec, scale)


def test_uniformscale_build_2d():
    scale = 2
    homo = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]])

    tr = UniformScale(scale, 2)
    assert_equal(tr.h_matrix, homo)


def test_uniformscale_build_3d():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    tr = UniformScale(scale, 3)

    assert(isinstance(tr, UniformScale))
    assert_equal(tr.h_matrix, homo)


@raises(ValueError)
def test_uniformscale_build_4d_raise_dimensionalityerror():
    UniformScale(1, 4)


def test_scale_build_2d_uniform_pass_dim():
    scale = 2
    ndim = 2
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale))


def test_scale_build_3d_uniform_pass_dim():
    scale = 2
    ndim = 3
    tr = Scale(scale, ndim)

    assert(isinstance(tr, UniformScale))


def test_scale_build_2d_nonuniform():
    scale = np.array([1, 2])
    tr = Scale(scale)

    assert(isinstance(tr, NonUniformScale))


def test_scale_build_2d_uniform_from_vec():
    scale = np.array([2, 2])
    tr = Scale(scale)

    assert(isinstance(tr, UniformScale))


@raises(ValueError)
def test_scale_zero_scale_raise_valuerror():
    Scale(np.array([1, 0]))


# Vectorizable interface tests

@raises(NotImplementedError)
def test_rotation3d_from_vector_raises_notimplementederror():
    Rotation.init_identity(3).from_vector(0)


@raises(NotImplementedError)
def test_rotation3d_as_vector_raises_notimplementederror():
    Rotation.init_identity(3).as_vector()


def test_affine_2d_n_parameters():
    homo = np.eye(3)
    t = Affine(homo)
    assert(t.n_parameters == 6)


def test_affine_2d_n_dims_output():
    homo = np.eye(3)
    t = Affine(homo)
    assert(t.n_dims_output == 2)


def test_affine_3d_n_parameters():
    homo = np.eye(4)
    t = Affine(homo)
    assert(t.n_parameters == 12)


def test_similarity_2d_n_parameters():
    homo = np.eye(3)
    t = Similarity(homo)
    assert(t.n_parameters == 4)


@raises(NotImplementedError)
def test_similarity_3d_n_parameters_raises_notimplementederror():
    homo = np.eye(4)
    t = Similarity(homo)
    # Raises exception
    t.n_parameters


def test_uniformscale2d_n_parameters():
    scale = 2
    t = UniformScale(scale, 2)
    assert(t.n_parameters == 1)


def test_uniformscale3d_n_parameters():
    scale = 2
    t = UniformScale(scale, 3)
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


# Test list construction is equivalent to ndarray construction

def test_translation_from_list():
    t_a = Translation([3, 4])
    t_b = Translation(np.array([3, 4]))
    assert(np.all(t_a.h_matrix == t_b.h_matrix))


def test_nonuniformscale_from_list():
    u_a = NonUniformScale([3, 2, 3])
    u_b = NonUniformScale(np.array([3, 2, 3]))
    assert(np.all(u_a.h_matrix == u_b.h_matrix))


# Test set_h_matrix is deprecated (and disabled)

@raises(NotImplementedError)
def test_homogenous_set_h_matrix_raises_notimplementederror():
    s = Homogeneous(np.eye(4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s.set_h_matrix(s.h_matrix)


def test_homogeneous_print():
    e = np.eye(3)
    h = Homogeneous(e)
    assert(str(h) == 'Homogeneous\n[[ 1.  0.  0.]\n [ 0.  1.  0.]'
                     '\n [ 0.  0.  1.]]')


def test_homogeneous_eye():
    e = np.eye(3)
    h = Homogeneous.init_identity(2)
    assert_allclose(e, h.h_matrix)


def test_homogeneous_has_true_inverse():
    h = Homogeneous.init_identity(2)
    assert h.has_true_inverse


def test_homogeneous_inverse():
    e = np.eye(3) * 2
    e[2, 2] = 1
    e_inv = np.eye(3) * 0.5
    e_inv[2, 2] = 1
    h = Homogeneous(e)
    assert_allclose(h.pseudoinverse().h_matrix, e_inv)


def test_homogeneous_apply():
    e = np.eye(3) * 2
    p = np.random.rand(10, 2)
    e[2, 2] = 1
    e[:2, -1] = [2, 3]
    h = Homogeneous(e)
    p_applied = h.apply(p)
    p_manual = p * 2 + np.array([2, 3])
    assert_allclose(p_applied, p_manual)


def test_homogeneous_apply_batched():
    e = np.eye(3) * 2
    p = np.random.rand(10, 2)
    e[2, 2] = 1
    e[:2, -1] = [2, 3]
    h = Homogeneous(e)
    p_applied = h.apply(p, batch_size=2)
    p_manual = p * 2 + np.array([2, 3])
    assert_allclose(p_applied, p_manual)


def test_homogeneous_as_vector():
    e = np.eye(3) * 2
    e[2, 2] = 1
    h = Homogeneous(e)
    assert_allclose(h.as_vector(), e.flatten())


def test_homogeneous_from_vector_inplace():
    h = Homogeneous(np.eye(3))
    e = np.eye(3) * 2
    e[2, 2] = 1
    h._from_vector_inplace(e.ravel())
    assert_allclose(h.h_matrix, e)
