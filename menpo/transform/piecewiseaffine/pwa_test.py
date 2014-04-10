import numpy as np
from numpy.testing import assert_equal
from menpo.transform.piecewiseaffine.base import DiscreteAffinePWA
from menpo.shape import PointCloud, TriMesh

src_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
tgt_points = np.array([[0, 0], [2, 0], [0, 2], [2, 3]])
trilist = np.array([[0, 1, 2], [1, 3, 2]])
src = TriMesh(src_points, trilist)
tgt = PointCloud(tgt_points)

a_affine = np.array(
    [[2.,  0.,  0.],
     [0.,  2.,  0.],
     [0.,  0.,  1.]])

b_affine = np.array(
    [[2.,  0.,  0.],
     [1.,  3., -1.],
     [0.,  0.,  1.]])


def test_pwa_discrete_affine_transforms():
    pwa = DiscreteAffinePWA(src, tgt)
    assert(len(pwa.transforms) == 2)
    assert_equal(pwa.transforms[0].h_matrix, a_affine)
    assert_equal(pwa.transforms[1].h_matrix, b_affine)
