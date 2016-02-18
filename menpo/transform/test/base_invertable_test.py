import numpy as np
from numpy.testing import assert_allclose

from menpo.base import Vectorizable
from menpo.transform.base import VInvertible


ones_vector = np.ones(3)


class MockedVInvertable(VInvertible, Vectorizable):
    def __init__(self):
        self.vector = ones_vector

    def _from_vector_inplace(self, vector):
        self.vector = vector

    def _as_vector(self):
        return self.vector

    def pseudoinverse(self):
        m = MockedVInvertable()
        m.vector = -self.vector
        return m

    def has_true_inverse(self):
        return True


def vinertable_pseudoinverse_test():
    v = MockedVInvertable()
    inv = v.pseudoinverse()
    assert_allclose(inv.vector, -ones_vector)


def vinertable_pseudoinverse_vector_test():
    v = MockedVInvertable()
    arr = np.array([1, 2])
    vec = v.pseudoinverse_vector(arr)
    assert_allclose(vec, -arr)
