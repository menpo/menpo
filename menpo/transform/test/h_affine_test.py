import numpy as np
from nose.tools import raises
from numpy.testing import assert_allclose

from menpo.transform import Affine, NonUniformScale


def test_affine_identity_2d():
    assert_allclose(Affine.identity(2).h_matrix, np.eye(3))


def test_affine_identity_3d():
    assert_allclose(Affine.identity(3).h_matrix, np.eye(4))


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
    x = np.array([[0, 1, 2],
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


def test_affine_2d_n_parameters():
    homo = np.eye(3)
    t = Affine(homo)
    assert (t.n_parameters == 6)


def test_affine_3d_n_parameters():
    homo = np.eye(4)
    t = Affine(homo)
    assert (t.n_parameters == 12)


@raises(ValueError)
def test_affine_non_square_h_matrix():
    homo = np.random.rand(4, 6)
    Affine(homo)


@raises(ValueError)
def test_affine_incorrect_bottom_row():
    homo = np.random.rand(4, 4)
    Affine(homo)


@raises(ValueError)
def test_affine_non_square_h_matrix():
    homo = np.random.rand(4, 6)
    Affine(homo)


def test_affine_compose_inplace_affine():
    a = Affine.identity(2)
    b = Affine.identity(2)
    a.compose_before_inplace(b)
    assert(np.all(a.h_matrix == b.h_matrix))


def test_affine_pseudoinverse():
    s = NonUniformScale([4, 3])
    inv_man = NonUniformScale([1./4, 1./3])
    b = Affine(s.h_matrix)
    i = b.pseudoinverse()
    assert_allclose(i.h_matrix, inv_man.h_matrix)
