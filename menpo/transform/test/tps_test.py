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


def test_tps_jacobian_manual_corner_value_check():
    tps = ThinPlateSplines(src, tgt)
    dW_dxy = tps.jacobian_source(square_sample_points)
    jacobian_image = dW_dxy.reshape(xx.shape + (4, 2))
    dwdx_corner = np.array(
        [[1., 0.99619633, 0.99231034, 0.98835616, 0.98434133],
         [0.99619633, 0.99240803, 0.98853625, 0.98459474, 0.98059178],
         [0.99231034, 0.98853625, 0.98468246, 0.98075813, 0.9767711],
         [0.98835616, 0.98459474, 0.98075813, 0.97685217, 0.97288316],
         [0.98434133, 0.98059178, 0.9767711, 0.97288316, 0.96893276]])
    assert_allclose(jacobian_image[:5, :5, 0, 0], dwdx_corner)
    assert_allclose(jacobian_image[:5, :5, 0, 1], dwdx_corner)


def test_tps_jacobian_unitary_x_y():
    tps = ThinPlateSplines(src, tgt)
    dW_dxy = tps.jacobian_source(square_sample_points)
    jacobian_image = dW_dxy.reshape(xx.shape + (4, 2))
    # both the x and y derivatives summed over all values should equal 1
    assert_allclose(jacobian_image.sum(axis=2), 1)


def test_tps_jacobian_manual_sample_a():
    tps = ThinPlateSplines(src, tgt_perturbed)
    dW_dxy = tps.jacobian_source(square_sample_points)
    onetwothreefour = np.array(
        [[0.78665966, 0.62374388],
         [0.09473867, -0.68910365],
         [0.68788235, 0.18713078],
         [-0.80552584, 1.11078411]])
    assert_allclose(dW_dxy[1234], onetwothreefour, rtol=10 ** -6)


def test_tps_jacobian_manual_sample_b():
    tps = ThinPlateSplines(src, tgt_perturbed)
    dW_dxy = tps.jacobian_source(square_sample_points)
    threesixfouronethree = np.array(
        [[0.67244171, -0.0098011],
         [-0.77028611, -1.19256858],
         [0.8296718, 0.95940495],
         [-0.03122042, 1.00073478]])
    assert_allclose(dW_dxy[36413], threesixfouronethree, atol=1e-5)


def test_tps_build_pseudoinverse():
    tps = ThinPlateSplines(src, tgt)
    tps_pinv = tps.pseudoinverse

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


def test_tps_jacobian_points():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    tps = ThinPlateSplines(src, tgt)
    result = tps.jacobian_points(src.points)

    assert_allclose(result, np.array([[[2, 0], [0., 2]],
                                      [[2, 0], [0, 2]],
                                      [[2, 0], [0, 2]],
                                      [[2, 0], [0, 2]]]), atol=10 ** -14)


def test_tps_jacobian_source():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    pts = np.array([[-0.1, -1.0], [-0.5, 1.0], [2.1, -2.5]])
    tps = ThinPlateSplines(src, tgt)
    result = tps.jacobian_source(pts)
    expected = np.array([[[1.10799034, 1.10799034], [-0.00799034, -0.00799034],
                          [0.89200966, 0.89200966], [0.00799034, 0.00799034]],
                         [[-0.0325033, -0.0325033], [1.5325033, 1.5325033],
                          [0.0325033, 0.0325033], [0.4674967, 0.4674967]],
                         [[0.03263195, 0.03263195], [-1.13263195, -1.13263195],
                          [3.46736805, 3.46736805],
                          [-0.36736805, -0.36736805]]])
    assert_allclose(result, expected, rtol=10 ** -6)


def test_tps_weight_points():
    src = PointCloud(np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]]))
    tgt = PointCloud(np.array([[-2.0, -2.0], [-2, 2], [2, -2], [2, 2]]))
    pts = np.array([[-0.1, -1.0], [-0.5, 1.0], [2.1, -2.5]])
    tps = ThinPlateSplines(src, tgt)
    result = tps.weight_points(pts)
    expected = np.array([[[0.55399517, 0.55399517], [-0.00399517, -0.00399517],
                          [0.44600483, 0.44600483], [0.00399517, 0.00399517]],
                         [[-0.01625165, -0.01625165], [0.76625165, 0.76625165],
                          [0.01625165, 0.01625165], [0.23374835, 0.23374835]],
                         [[0.01631597, 0.01631597], [-0.56631597, -0.56631597],
                          [1.73368403, 1.73368403],
                          [-0.18368403, -0.18368403]]])
    assert_allclose(result, expected, rtol=10 ** -6)
