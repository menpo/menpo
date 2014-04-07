import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

from menpo.transform import (Homogeneous, Scale, TransformChain,
                             NonUniformScale,
                            UniformScale, Translation, Similarity)


def homog_compose_before_scale_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = Scale([3, 4])
    res = homog.compose_before(s)
    assert(isinstance(res, Homogeneous))
    assert_allclose(res.h_matrix, np.array([[0, 3, 0],
                                            [4, 0, 0],
                                            [0, 0, 1]]))


def homog_compose_after_scale_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = Scale([3, 4])
    res = homog.compose_after(s)
    assert(isinstance(res, Homogeneous))
    assert_allclose(res.h_matrix, np.array([[0, 4, 0],
                                            [3, 0, 0],
                                            [0, 0, 1]]))


def nonuniformscale_compose_after_homog_test():
    # can't do this inplace - so should just give transform chain
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = NonUniformScale([3, 4])
    res = s.compose_after(homog)
    assert(type(res) == Homogeneous)


@raises(ValueError)
def scale_compose_after_inplace_homog_test():
    # can't do this inplace - so should just give transform chain
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = Scale([3, 4])
    s.compose_after_inplace(homog)


def homog_compose_after_inplace_scale_test():
    # this should be fine
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = Scale([3, 4])
    homog.compose_after_inplace(s)
    assert_allclose(homog.h_matrix, np.array([[0, 4, 0],
                                              [3, 0, 0],
                                              [0, 0, 1]]))


def uniformscale_compose_after_translation_test():
    t = Translation([3, 4])
    s = UniformScale(2, 2)
    res = s.compose_after(t)
    assert(type(res) == Similarity)


def translation_compose_after_uniformscale_test():
    t = Translation([3, 4])
    s = UniformScale(2, 2)
    res = t.compose_after(s)
    assert(type(res) == Similarity)
