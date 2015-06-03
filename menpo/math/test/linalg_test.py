from nose.tools import raises
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from menpo.math import (dot_inplace_left, dot_inplace_right, as_matrix,
                        from_matrix)
from menpo.image import MaskedImage


n_big = 9182
k = 100
n_small = 99

n_images = 5
image_shape = (10, 10)
mask = np.zeros(image_shape, dtype=np.bool)
mask[:2] = True
template = MaskedImage.init_blank(image_shape, mask=mask)

matrix = np.random.random([n_images, 20])

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


def test_as_matrix_list():
    data = as_matrix([template.copy() for _ in range(n_images)])
    # Two rows of the mask are True (10 * 2 = 20)
    assert_equal(data.shape, (n_images, 20))


def test_as_matrix_generator():
    data = as_matrix((template.copy() for _ in range(n_images)),
                     length=n_images)
    # Two rows of the mask are True (10 * 2 = 20)
    assert_equal(data.shape, (n_images, 20))


def test_as_matrix_short_length():
    data = as_matrix((template.copy() for _ in range(n_images)), length=1)
    # Two rows of the mask are True (10 * 2 = 20)
    assert_equal(data.shape, (1, 20))


@raises(ValueError)
def test_as_matrix_long_length_raises_value_error():
    as_matrix((template.copy() for _ in range(4)), length=5)


def test_as_matrix_return_template():
    data, t = as_matrix((template.copy() for _ in range(n_images)),
                        length=1, return_template=True)
    # Two rows of the mask are True (10 * 2 = 20)
    assert_equal(data.shape, (1, 20))
    assert_equal(t.shape, image_shape)


def test_from_matrix():
    images = from_matrix(matrix, template)
    assert isinstance(next(images), MaskedImage)
