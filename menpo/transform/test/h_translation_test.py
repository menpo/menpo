import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from nose.tools import raises

from menpo.transform import Translation


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


def test_translation_2d_from_vector():
    params = np.array([1, 2])
    homo = np.array([[1, 0, params[0]],
                     [0, 1, params[1]],
                     [0, 0, 1]])

    tr = Translation.identity(2).from_vector(params)

    assert_almost_equal(tr.h_matrix, homo)


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

    tr = Translation.identity(3).from_vector(params)

    assert_almost_equal(tr.h_matrix, homo)


def test_translation_3d_as_vector():
    params = np.array([1, 2, 3])
    vec = Translation(params).as_vector()
    assert_allclose(vec, params)


def test_translation_2d_n_parameters():
    trans = np.array([1, 2])
    t = Translation(trans)
    assert (t.n_parameters == 2)


def test_translation_3d_n_parameters():
    trans = np.array([1, 2, 3])
    t = Translation(trans)
    assert (t.n_parameters == 3)


@raises(NotImplementedError)
def test_translation_set_h_matrix_raises_notimplementederror():
    t = Translation([3, 4])
    t.set_h_matrix(t.h_matrix)


def test_translation_from_list():
    t_a = Translation([3, 4])
    t_b = Translation(np.array([3, 4]))
    assert (np.all(t_a.h_matrix == t_b.h_matrix))


def test_translation_identity_2d():
    assert_allclose(Translation.identity(2).h_matrix, np.eye(3))


def test_translation_identity_3d():
    assert_allclose(Translation.identity(3).h_matrix, np.eye(4))


def test_translation_decompose_optional():
    t = Translation.identity(2)
    d = t.decompose()
    assert np.all(d[0].h_matrix == t.h_matrix)
