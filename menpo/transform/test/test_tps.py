import numpy as np
from numpy.testing import assert_allclose

from menpo.transform.thinplatesplines import ThinPlateSplines
from menpo.shape import PointCloud


square_src_landmarks = np.array([[-1.0, -1.0],
                                 [-1, 1],
                                 [1, -1],
                                 [1, 1]])

square_tgt_landmarks = np.array([[-1.0, -1.0],
                                 [-1, 1],
                                 [1, -1],
                                 [1, 1]])

perturbed_tgt_landmarks = np.array([[-0.6, -1.3],
                                    [-0.8, 1.2],
                                    [0.7, -0.8],
                                    [1.3, 0.5]])

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y)
square_sample_points = np.array([xx.flatten(1), yy.flatten(1)]).T

src = PointCloud(square_src_landmarks)
tgt = PointCloud(square_tgt_landmarks)
tgt_perturbed = PointCloud(perturbed_tgt_landmarks)


def test_tps_maps_src_to_tgt():
    tps = ThinPlateSplines(src, tgt_perturbed)
    assert_allclose(tps.apply(square_src_landmarks), perturbed_tgt_landmarks)


def test_tps_n_dims():
    tps = ThinPlateSplines(src, tgt_perturbed)
    assert(tps.n_dims == 2)


def test_tps_build_pseudoinverse():
    tps = ThinPlateSplines(src, tgt)
    tps_pinv = tps.pseudoinverse()

    assert (tps.source == tps_pinv.target)
    assert (tps.target == tps_pinv.source)
    # TODO: test the kernel


def test_tps_apply():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    pts = PointCloud(np.array([[-0.1, -1.0], [-0.5, 1.0], [2.1, -2.5]]))
    tps = ThinPlateSplines(src, tgt)
    result = tps.apply(pts)
    expected = np.array([[-0.2, -2.], [-1., 2.], [4.2, -5.]])
    assert_allclose(result.points, expected)


def test_tps_apply_batched():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    pts = PointCloud(np.array([[-0.1, -1.0], [-0.5, 1.0], [2.1, -2.5]]))
    tps = ThinPlateSplines(src, tgt)
    result = tps.apply(pts, batch_size=2)
    expected = np.array([[-0.2, -2.], [-1., 2.], [4.2, -5.]])
    assert_allclose(result.points, expected)
