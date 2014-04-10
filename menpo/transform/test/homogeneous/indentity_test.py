import numpy as np
from numpy.testing import assert_allclose
from menpo.transform import (Affine, NonUniformScale, Similarity,
                             UniformScale, Rotation, Translation)


def test_affine_identity_2d():
    assert_allclose(Affine.identity(2).h_matrix, np.eye(3))


def test_affine_identity_3d():
    assert_allclose(Affine.identity(3).h_matrix, np.eye(4))


def test_nonuniformscale_identity_2d():
    assert_allclose(NonUniformScale.identity(2).h_matrix, np.eye(3))


def test_nonuniformscale_identity_3d():
    assert_allclose(NonUniformScale.identity(3).h_matrix, np.eye(4))

def test_similarity_identity_2d():
    assert_allclose(Similarity.identity(2).h_matrix,
                    np.eye(3))


def test_similarity_identity_3d():
    assert_allclose(Similarity.identity(3).h_matrix,
                    np.eye(4))


def test_uniformscale_identity_2d():
    assert_allclose(UniformScale.identity(2).h_matrix, np.eye(3))


def test_uniformscale_identity_3d():
    assert_allclose(UniformScale.identity(3).h_matrix, np.eye(4))


def test_rotation2d_identity():
    assert_allclose(Rotation.identity(2).h_matrix, np.eye(3))


def test_rotation3d_identity():
    assert_allclose(Rotation.identity(3).h_matrix, np.eye(4))


def test_translation_identity_2d():
    assert_allclose(Translation.identity(2).h_matrix, np.eye(3))


def test_translation_identity_3d():
    assert_allclose(Translation.identity(3).h_matrix, np.eye(4))
