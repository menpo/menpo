from numpy.testing import assert_allclose
import numpy as np
from menpo.transform import R2LogR2RBF, R2LogRRBF

centers = np.array([[-1.0, -1.0], [-1, 1], [1, -1], [1, 1]])
points = np.array([[-0.4, -1.5], [-0.1, 1.1], [0.1, -2], [2.3, 0.3]])


def test_rbf_r2logr2_apply():
    result = R2LogR2RBF(centers).apply(points)
    expected = np.array([[-0.30152076, 12.48353795, 1.75251346, 17.2849475],
                         [8.62603644, -0.16272977, 9.70198395, 0.24259805],
                         [1.75251346, 23.72158352, 1.07392159, 22.4001763],
                         [31.8539218, 27.67453754, 4.1164199, 1.69892823]])
    assert_allclose(result, expected)


def test_rbf_r2logr_apply():
    result = R2LogRRBF(centers).apply(points)
    expected = np.array([[-0.15076038, 6.24176898, 0.87625673, 8.64247375],
                         [4.31301822, -0.08136488, 4.85099198, 0.12129902],
                         [0.87625673, 11.86079176, 0.53696079, 11.20008815],
                         [15.9269609, 13.83726877, 2.05820995, 0.84946412]])
    assert_allclose(result, expected)
