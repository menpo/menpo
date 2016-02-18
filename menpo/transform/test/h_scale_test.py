import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
from nose.tools import raises

from menpo.transform import UniformScale, NonUniformScale, Scale


def test_nonuniformscale_from_list():
    u_a = NonUniformScale([3, 2, 3])
    u_b = NonUniformScale(np.array([3, 2, 3]))
    assert (np.all(u_a.h_matrix == u_b.h_matrix))


def test_uniformscale2d_n_parameters():
    scale = 2
    t = UniformScale(scale, 2)
    assert (t.n_parameters == 1)


def test_uniformscale3d_n_parameters():
    scale = 2
    t = UniformScale(scale, 3)
    assert (t.n_parameters == 1)


def test_nonuniformscale_2d_n_parameters():
    scale = np.array([1, 2])
    t = NonUniformScale(scale)
    assert (t.n_parameters == 2)


def test_uniformscale2d_update_from_vector():
    # make a uniform scale of 1, 2 dimensional
    uniform_scale = UniformScale(1, 2)
    new_scale = 2
    homo = np.array([[new_scale, 0, 0],
                     [0, new_scale, 0],
                     [0, 0, 1]])

    uniform_scale._from_vector_inplace(new_scale)
    assert_almost_equal(uniform_scale.h_matrix, homo)


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

    assert_almost_equal(tr.h_matrix, homo)


def test_nonuniformscale2d_update_from_vector():
    scale = np.array([3, 4])
    homo = np.array([[scale[0], 0, 0],
                     [0, scale[1], 0],
                     [0, 0, 1]])
    tr = NonUniformScale(np.array([1, 2]))
    tr._from_vector_inplace(scale)
    assert_almost_equal(tr.h_matrix, homo)


def test_nonuniformscale2d_as_vector():
    scale = np.array([1, 2])
    vec = NonUniformScale(scale).as_vector()
    assert_allclose(vec, scale)


def test_scale_2d_pseudoinverse():
    scale1 = 0.5
    scale2 = 4.0
    homo = np.array([[scale1,      0, 0],
                     [     0, scale2, 0],
                     [     0,      0, 1]])

    tr = NonUniformScale([1/scale1, 1/scale2])
    assert_almost_equal(tr.pseudoinverse().h_matrix, homo)


def test_uniformscale3d_from_vector():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    uniform_scale = UniformScale(1, 3)
    tr = uniform_scale.from_vector(scale)
    assert_almost_equal(tr.h_matrix, homo)


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
    assert_almost_equal(tr.h_matrix, homo)


def test_uniformscale_build_3d():
    scale = 2
    homo = np.array([[scale, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, scale, 0],
                     [0, 0, 0, 1]])

    tr = UniformScale(scale, 3)

    assert (isinstance(tr, UniformScale))
    assert_almost_equal(tr.h_matrix, homo)


@raises(ValueError)
def test_uniformscale_build_4d_raise_dimensionalityerror():
    UniformScale(1, 4)


def test_uniformscale_2d_pseudoinverse():
    scale = 0.5
    homo = np.array([[scale, 0, 0],
                     [0, scale, 0],
                     [0, 0, 1]])

    tr = UniformScale(2, 2)
    assert_almost_equal(tr.pseudoinverse().h_matrix, homo)


def test_scale_build_2d_uniform_pass_dim():
    scale = 2
    ndim = 2
    tr = Scale(scale, ndim)

    assert (isinstance(tr, UniformScale))


def test_scale_build_3d_uniform_pass_dim():
    scale = 2
    ndim = 3
    tr = Scale(scale, ndim)

    assert (isinstance(tr, UniformScale))


def test_scale_build_2d_nonuniform():
    scale = np.array([1, 2])
    tr = Scale(scale)

    assert (isinstance(tr, NonUniformScale))


def test_scale_build_2d_uniform_from_vec():
    scale = np.array([2, 2])
    tr = Scale(scale)

    assert (isinstance(tr, UniformScale))


@raises(ValueError)
def test_scale_zero_scale_raise_valuerror():
    Scale(np.array([1, 0]))


def test_uniformscale_identity_2d():
    assert_allclose(UniformScale.init_identity(2).h_matrix, np.eye(3))


def test_uniformscale_identity_3d():
    assert_allclose(UniformScale.init_identity(3).h_matrix, np.eye(4))


def test_nonuniformscale_identity_2d():
    assert_allclose(NonUniformScale.init_identity(2).h_matrix, np.eye(3))


def test_nonuniformscale_identity_3d():
    assert_allclose(NonUniformScale.init_identity(3).h_matrix, np.eye(4))
