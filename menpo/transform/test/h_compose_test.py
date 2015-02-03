import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises
from menpo.shape import PointCloud

from menpo.transform import (Homogeneous, Scale, TransformChain,
                             NonUniformScale,
                             UniformScale, Translation, Similarity, Rotation,
                             AlignmentUniformScale, AlignmentRotation)

# NON-INPLACE COMPOSE


# 1a. Homogenous before/after with subclasses. All should promote to Homogeneous

def homog_compose_before_nonuniformscale_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = NonUniformScale([3, 4])
    res = homog.compose_before(s)
    assert(type(res) == Homogeneous)
    assert_allclose(res.h_matrix, np.array([[0, 3, 0],
                                            [4, 0, 0],
                                            [0, 0, 1]]))


def homog_compose_after_uniformscale_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    s = UniformScale(3, 2)
    res = homog.compose_after(s)
    assert(type(res) == Homogeneous)
    assert_allclose(res.h_matrix, np.array([[0, 3, 0],
                                            [3, 0, 0],
                                            [0, 0, 1]]))


def rotation_compose_before_homog_test():
    # can't do this inplace - so should just give transform chain
    rotation = Rotation(np.array([[1, 0],
                                  [0, 1]]))
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    res = rotation.compose_before(homog)
    assert(type(res) == Homogeneous)


def translation_compose_after_homog_test():
    # can't do this inplace - so should just give transform chain
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    t = Translation([3, 4])
    res = t.compose_after(homog)
    assert(type(res) == Homogeneous)


# 1b. Homogeneous composed with Alignment subclasses. Should loose alignment
# traits.

def homog_compose_before_alignment_nonuniformscale_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    scale = UniformScale(2.5, 2)
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    target = scale.apply(source)
    # estimate the transform from source and target
    s = AlignmentUniformScale(source, target)
    res = homog.compose_before(s)
    assert(type(res) == Homogeneous)


def homog_compose_after_alignment_rotation_test():
    homog = Homogeneous(np.array([[0, 1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]))
    source = PointCloud(np.array([[0, 1],
                                  [1, 1],
                                  [-1, -5],
                                  [3, -5]]))
    r = AlignmentRotation(source, source)
    res = homog.compose_after(r)
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
