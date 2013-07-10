import numpy as np
from numpy.testing import assert_allclose
from pybug.align.nonrigid.tps import TPS
from pybug.exceptions import DimensionalityError
from nose.tools import raises


# @raises(DimensionalityError)
# def test_tps_map_non_2d():
#     t_vec = np.array([1])
#     Translation(t_vec)


def test_tps_maps_src_to_tgt():
    src_landmarks = np.array([[0, 1.0],
                              [-1, 0.0],
                              [0, -1.0],
                              [1, 0.0]])

    tgt_landmarks = np.array([[0, 0.75],
                              [-1, 0.25],
                              [0, -1.25],
                              [1, 0.25]])

    tps = TPS(src_landmarks, tgt_landmarks)
    assert_allclose(tps.transform.apply(src_landmarks), tgt_landmarks)