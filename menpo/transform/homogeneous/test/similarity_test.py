import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from nose.tools import raises

from menpo.transform import Similarity


sim_jac_solution2d = np.array([[[0., 0.],
                                [0., 0.],
                                [1., 0.],
                                [0., 1.]],
                               [[0., 1.],
                                [-1., 0.],
                                [1., 0.],
                                [0., 1.]],
                               [[0., 2.],
                                [-2., 0.],
                                [1., 0.],
                                [0., 1.]],
                               [[1., 0.],
                                [0., 1.],
                                [1., 0.],
                                [0., 1.]],
                               [[1., 1.],
                                [-1., 1.],
                                [1., 0.],
                                [0., 1.]],
                               [[1., 2.],
                                [-2., 1.],
                                [1., 0.],
                                [0., 1.]]])


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


def test_similarity_jacobian_2d():
    params = np.ones(4)
    t = Similarity.identity(2).from_vector(params)
    explicit_pixel_locations = np.array(
        [[0, 0],
         [0, 1],
         [0, 2],
         [1, 0],
         [1, 1],
         [1, 2]])
    dW_dp = t.d_dp(explicit_pixel_locations)
    assert_almost_equal(dW_dp, sim_jac_solution2d)


@raises(ValueError)
def test_similarity_jacobian_3d_raises_dimensionalityerror():
    t = Similarity(np.eye(4))
    t.d_dp(np.ones([2, 3]))


@raises(ValueError)
def test_similarity_2d_points_raises_dimensionalityerror():
    params = np.ones(4)
    t = Similarity.identity(2).from_vector(params)
    t.d_dp(np.ones([2, 3]))


def test_similarity_2d_from_vector():
    params = np.array([0.2, 0.1, 1, 2])
    homo = np.array([[params[0] + 1, -params[1], params[2]],
                     [params[1], params[0] + 1, params[3]],
                     [0, 0, 1]])

    sim = Similarity.identity(2).from_vector(params)

    assert_almost_equal(sim.h_matrix, homo)


def test_similarity_2d_as_vector():
    params = np.array([0.2, 0.1, 1.0, 2.0])
    homo = np.array([[params[0] + 1.0, -params[1], params[2]],
                     [params[1], params[0] + 1.0, params[3]],
                     [0.0, 0.0, 1.0]])

    vec = Similarity(homo).as_vector()

    assert_allclose(vec, params)


def test_similarity_2d_n_parameters():
    homo = np.eye(3)
    t = Similarity(homo)
    assert (t.n_parameters == 4)


@raises(NotImplementedError)
def test_similarity_3d_n_parameters_raises_notimplementederror():
    homo = np.eye(4)
    t = Similarity(homo)
    # Raises exception
    t.n_parameters


@raises(NotImplementedError)
def test_similarity_set_h_matrix_raises_notimplementederror():
    s = Similarity(np.eye(3))
    s.set_h_matrix(s.h_matrix)


def test_similarity_identity_2d():
    assert_allclose(Similarity.identity(2).h_matrix,
                    np.eye(3))


def test_similarity_identity_3d():
    assert_allclose(Similarity.identity(3).h_matrix,
                    np.eye(4))
