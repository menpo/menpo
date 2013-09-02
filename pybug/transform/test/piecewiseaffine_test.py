import numpy as np
from numpy.testing import assert_allclose, assert_equal
from pybug.transform.piecewiseaffine import PWATransform


a = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
b = np.array([[0, 0], [2, 0], [0, 2], [2, 3]])
tl = np.array([[0, 1, 2], [1, 3, 2]])

a_affine = np.array(
    [[2.,  0.,  0.],
     [0.,  2.,  0.],
     [0.,  0.,  1.]])

b_affine = np.array(
    [[2.,  0.,  0.],
     [1.,  3., -1.],
     [0.,  0.,  1.]])


def test_affine_transforms():
    pwa = PWATransform(a, b, tl)
    assert(len(pwa.transforms) == 2)
    assert_equal(pwa.transforms[0].homogeneous_matrix, a_affine)
    assert_equal(pwa.transforms[1].homogeneous_matrix, b_affine)


def test_pwa_n_parameters():
    pwa = PWATransform(a, b, tl)
    assert(pwa.n_parameters == 12)