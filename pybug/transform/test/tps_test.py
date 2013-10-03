import numpy as np
from numpy.testing import assert_allclose
from pybug.transform.tps import TPS
from pybug.shape import PointCloud

square_src_landmarks = np.array([[-1.0, -1.0],
                                 [-1,  1],
                                 [1, -1],
                                 [1,  1]])

square_tgt_landmarks = np.array([[-1.0, -1.0],
                                 [-1,  1],
                                 [1, -1],
                                 [1,  1]])

perturbed_tgt_landmarks = np.array([[-0.6, -1.3],
                                    [-0.8,  1.2],
                                    [0.7, -0.8],
                                    [1.3,  0.5]])

x = np.arange(-1, 1, 0.01)
y = np.arange(-1, 1, 0.01)
xx, yy = np.meshgrid(x, y)
square_sample_points = np.array([xx.flatten(1), yy.flatten(1)]).T


src = PointCloud(square_src_landmarks)
tgt = PointCloud(square_tgt_landmarks)
tgt_perturbed = PointCloud(perturbed_tgt_landmarks)


def test_tps_maps_src_to_tgt():
    tps = TPS(src, tgt_perturbed)
    assert_allclose(tps.apply(square_src_landmarks), perturbed_tgt_landmarks)


def test_tps_jacobian_manual_corner_value_check():
    tps = TPS(src, tgt)
    dW_dxy = tps.jacobian_source(square_sample_points)
    jacobian_image = dW_dxy.reshape(xx.shape + (4, 2))
    dwdx_corner = np.array(
        [[1.,  0.99619633,  0.99231034,  0.98835616, 0.98434133],
         [0.99619633,  0.99240803,  0.98853625,  0.98459474, 0.98059178],
         [0.99231034,  0.98853625,  0.98468246,  0.98075813,  0.9767711],
         [0.98835616,  0.98459474,  0.98075813,  0.97685217,  0.97288316],
         [0.98434133,  0.98059178,  0.9767711,   0.97288316,  0.96893276]])
    assert_allclose(jacobian_image[:5, :5, 0, 0], dwdx_corner)
    assert_allclose(jacobian_image[:5, :5, 0, 1], dwdx_corner)


def test_tps_jacobian_unitary_x_y():
    tps = TPS(src, tgt)
    dW_dxy = tps.jacobian_source(square_sample_points)
    jacobian_image = dW_dxy.reshape(xx.shape + (4, 2))
    # both the x and y derivatives summed over all values should equal 1
    assert_allclose(jacobian_image.sum(axis=2), 1)


def test_tps_jacobian_manual_sample_a():
    tps = TPS(src, tgt_perturbed)
    dW_dxy = tps.jacobian_source(square_sample_points)
    onetwothreefour = np.array(
        [[0.78665966,  0.62374388],
         [0.09473867, -0.68910365],
         [0.68788235,  0.18713078],
         [-0.80552584,  1.11078411]])
    assert_allclose(dW_dxy[1234], onetwothreefour)


def test_tps_jacobian_manual_sample_b():
    tps = TPS(src, tgt_perturbed)
    dW_dxy = tps.jacobian_source(square_sample_points)
    threesixfouronethree = np.array(
        [[0.67244171, -0.0098011 ],
         [-0.77028611, -1.19256858],
         [0.8296718, 0.95940495],
         [-0.03122042,  1.00073478]])
    assert_allclose(dW_dxy[36413], threesixfouronethree, atol=1e-5)


def test_tps_n_parameters():
    tps = TPS(src, tgt_perturbed)
    assert(tps.n_parameters == 14)
