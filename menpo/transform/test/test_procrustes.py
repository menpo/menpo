import numpy as np
from numpy.testing import assert_allclose

from menpo.shape import PointCloud
from menpo.transform import GeneralizedProcrustesAnalysis


def test_procrustes_no_target():
    # square
    src_1 = PointCloud(np.array([[1.0, 1.0], [1.0, -1.0],
                                 [-1.0, -1.0], [-1.0, 1.0]]))
    # rhombus
    src_2 = PointCloud(np.array([[2.0, 0.0], [4.0, 2.0],
                                 [6.0, 0.0], [4.0, -2.0]]))
    # trapezoid
    src_3 = PointCloud(np.array([[-0.5, -1.5], [2.5, -1.5],
                                 [2.8, -2.5], [-0.2, -2.5]]))
    gpa = GeneralizedProcrustesAnalysis([src_1, src_2, src_3],
                                        allow_mirror=True)
    aligned_1 = gpa.transforms[0].apply(gpa.sources[0])
    aligned_2 = gpa.transforms[1].apply(gpa.sources[1])
    aligned_3 = gpa.transforms[2].apply(gpa.sources[2])
    assert(gpa.converged is True)
    assert(gpa.n_iterations == 5)
    assert(gpa.n_sources == 3)
    assert(np.round(gpa.initial_target_scale * 100) == 195.)
    assert(np.round(gpa.mean_alignment_error() * 10) == 4.)
    assert_allclose(np.around(aligned_1.points, decimals=1),
                    np.around(aligned_2.points, decimals=1))
    res_3 = np.array([[0.7, -0.3], [2.6, -0.4], [2.7, -1.0], [0.9, -0.9]])
    assert_allclose(np.around(aligned_3.points, decimals=1), res_3)
    assert_allclose(gpa.target.points, gpa.mean_aligned_shape().points)


def test_procrustes_with_target():
    # square
    src_1 = PointCloud(np.array([[1.0, 1.0], [1.0, -1.0],
                                 [-1.0, -1.0], [-1.0, 1.0]]))
    # trapezoid
    src_2 = PointCloud(np.array([[-0.5, -1.5], [2.5, -1.5],
                                 [2.8, -2.5], [-0.2, -2.5]]))
    # rhombus as target
    src_trg = PointCloud(np.array([[2.0, 0.0], [4.0, 2.0],
                                   [6.0, 0.0], [4.0, -2.0]]))
    gpa = GeneralizedProcrustesAnalysis([src_1, src_2], target=src_trg,
                                        allow_mirror=True)
    aligned_1 = gpa.transforms[0].apply(gpa.sources[0])
    aligned_2 = gpa.transforms[1].apply(gpa.sources[1])
    assert(gpa.converged is True)
    assert(gpa.n_iterations == 2)
    assert(gpa.n_sources == 2)
    assert(gpa.initial_target_scale == 4.)
    assert(np.round(gpa.mean_alignment_error() * 100) == 93.)
    assert_allclose(np.around(aligned_1.points, decimals=1),
                    np.around(src_trg.points, decimals=1))
    res_2 = np.array([[2.0, -0.9], [4.9, 1.6], [6.0, 0.9], [3.1, -1.6]])
    assert_allclose(np.around(aligned_2.points, decimals=1), res_2)
    mean = np.array([[2.0, -0.5], [4.5, 1.8], [6.0, 0.5], [3.5, -1.8]])
    assert_allclose(np.around(gpa.mean_aligned_shape().points, decimals=1),
                    mean)
