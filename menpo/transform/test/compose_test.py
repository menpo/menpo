import numpy as np
from numpy.testing import assert_allclose
from nose.tools import raises

from menpo.transform.tps import TPS
from menpo.transform.piecewiseaffine import PiecewiseAffineTransform
from menpo.shape import PointCloud, TriMesh
from menpo.transform.affine import Translation, Scale
from menpo.transform.homogeneous import HomogeneousTransform
from menpo.transform.base import TransformChain


def chain_tps_before_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps_one = TPS(a, b)
    tps_two = TPS(b, a)
    chain = tps_one.compose_before(tps_two)
    assert(isinstance(chain, TransformChain))
    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain.apply(points)
    manual_res = tps_two.apply(tps_one.apply(points))
    assert (np.all(chain_res.points == manual_res.points))


def chain_tps_after_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps_one = TPS(a, b)
    tps_two = TPS(b, a)
    chain = tps_one.compose_after(tps_two)
    assert(isinstance(chain, TransformChain))
    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain.apply(points)
    manual_res = tps_one.apply(tps_two.apply(points))
    assert (np.all(chain_res.points == manual_res.points))


def chain_pwa_before_tps_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffineTransform(a_tm, b)
    tps = TPS(b, a_tm)
    chain = pwa.compose_before(tps)
    assert(isinstance(chain, TransformChain))


def chain_pwa_after_tps_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffineTransform(a_tm, b)
    tps = TPS(b, a_tm)
    chain = pwa.compose_after(tps)
    assert(isinstance(chain, TransformChain))


def chain_tps_before_pwa_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffineTransform(a_tm, b)
    tps = TPS(b, a_tm)
    chain = tps.compose_before(pwa)
    assert(isinstance(chain, TransformChain))


def chain_tps_after_pwa_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffineTransform(a_tm, b)
    tps = TPS(b, a_tm)
    chain = tps.compose_after(pwa)
    assert(isinstance(chain, TransformChain))


def compose_tps_after_translation_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    t = Translation([3, 4])
    tps = TPS(a, b)
    chain = tps.compose_after(t)
    assert(isinstance(chain, TransformChain))


def manual_no_op_chain_test():
    points = PointCloud(np.random.random([10, 2]))
    t = Translation([3, 4])
    chain = TransformChain([t, t.pseudoinverse])
    points_applied = chain.apply(points)
    assert(np.allclose(points_applied.points, points.points))


def chain_compose_before_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps = TPS(a, b)

    t = Translation([3, 4])
    s = Scale([4, 2])
    chain = TransformChain([t, s])
    chain_mod = chain.compose_before(tps)

    points = PointCloud(np.random.random([10, 2]))

    manual_res = tps.apply(s.apply(t.apply(points)))
    chain_res = chain_mod.apply(points)
    assert(np.all(manual_res.points == chain_res.points))


def chain_compose_after_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps = TPS(a, b)

    t = Translation([3, 4])
    s = Scale([4, 2])
    chain = TransformChain([t, s])
    chain_mod = chain.compose_after(tps)

    points = PointCloud(np.random.random([10, 2]))

    manual_res = s.apply(t.apply(tps.apply(points)))
    chain_res = chain_mod.apply(points)
    assert(np.all(manual_res.points == chain_res.points))


def chain_compose_before_inplace_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps = TPS(a, b)

    t = Translation([3, 4])
    s = Scale([4, 2])
    chain = TransformChain([t, s])
    chain.compose_before_inplace(tps)

    points = PointCloud(np.random.random([10, 2]))

    manual_res = tps.apply(s.apply(t.apply(points)))
    chain_res = chain.apply(points)
    assert(np.all(manual_res.points == chain_res.points))


def chain_compose_after_inplace_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps = TPS(a, b)

    t = Translation([3, 4])
    s = Scale([4, 2])
    chain = TransformChain([t, s])
    chain.compose_after_inplace(tps)

    points = PointCloud(np.random.random([10, 2]))

    manual_res = s.apply(t.apply(tps.apply(points)))
    chain_res = chain.apply(points)
    assert(np.all(manual_res.points == chain_res.points))


def chain_compose_after_inplace_chain_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))

    t = Translation([3, 4])
    s = Scale([4, 2])
    chain_1 = TransformChain([t, s])
    chain_2 = TransformChain([s.pseudoinverse, t.pseudoinverse])
    chain_1.compose_before_inplace(chain_2)

    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain_1.apply(points)
    assert(np.allclose(points.points, chain_res.points))


def homog_compose_before_scale_test():
    homog = HomogeneousTransform(np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    s = Scale([3, 4])
    res = homog.compose_before(s)
    assert(isinstance(res, HomogeneousTransform))
    assert_allclose(res.h_matrix, np.array([[0, 3, 0],
                                            [4, 0, 0],
                                            [0, 0, 1]]))


def homog_compose_after_scale_test():
    homog = HomogeneousTransform(np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    s = Scale([3, 4])
    res = homog.compose_after(s)
    assert(isinstance(res, HomogeneousTransform))
    assert_allclose(res.h_matrix, np.array([[0, 4, 0],
                                            [3, 0, 0],
                                            [0, 0, 1]]))


def scale_compose_after_homog_test():
    # can't do this inplace - so should just give transform chain
    homog = HomogeneousTransform(np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    s = Scale([3, 4])
    res = s.compose_after(homog)
    assert(isinstance(res, TransformChain))


@raises(ValueError)
def scale_compose_after_inplace_homog_test():
    # can't do this inplace - so should just give transform chain
    homog = HomogeneousTransform(np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    s = Scale([3, 4])
    s.compose_after_inplace(homog)


def homog_compose_after_inplace_scale_test():
    # this should be fine
    homog = HomogeneousTransform(np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]]))
    s = Scale([3, 4])
    homog.compose_after_inplace(s)
    assert_allclose(homog.h_matrix, np.array([[0, 4, 0],
                                              [3, 0, 0],
                                              [0, 0, 1]]))
