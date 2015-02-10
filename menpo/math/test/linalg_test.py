from nose.tools import raises
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from menpo.math import dot_inplace_left, dot_inplace_right
n_big = 9182
k = 100
n_small = 99

a_l = np.random.rand(n_big, k)
b_l = np.random.rand(k, n_small)

a_r = np.ascontiguousarray(b_l.T)
b_r = np.ascontiguousarray(a_l.T)

gt_l = a_l.dot(b_l)
gt_r = a_r.dot(b_r)


def test_dot_inplace_left():
    a_l_tmp = a_l.copy()
    b_l_tmp = b_l.copy()
    left_result = dot_inplace_left(a_l_tmp, b_l_tmp)
    assert_allclose(left_result, gt_l)
    assert_equal(left_result, a_l_tmp[:, :n_small])
    assert_equal(b_l_tmp, b_l)


def test_dot_inplace_right():
    a_r_tmp = a_r.copy()
    b_r_tmp = b_r.copy()
    right_result = dot_inplace_right(a_r_tmp, b_r_tmp)
    assert_allclose(right_result, gt_r)
    assert_equal(right_result, b_r_tmp[:n_small])
    assert_equal(a_r_tmp, a_r)


@raises(ValueError)
def test_dot_inplace_left_n_small_too_big_raises_value_error():
    a = np.zeros((10000, 100))
    b = np.zeros((100, 101))
    dot_inplace_left(a, b)


@raises(ValueError)
def test_dot_inplace_right_n_small_too_big_raises_value_error():
    a = np.zeros((101, 100))
    b = np.zeros((100, 10000))
    dot_inplace_right(a, b)
