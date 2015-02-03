import numpy as np

from menpo.shape import PointCloud, TriMesh

from menpo.transform import TransformChain, Translation, Scale
from menpo.transform.thinplatesplines import ThinPlateSplines
from menpo.transform.piecewiseaffine import PiecewiseAffine


def chain_tps_before_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps_one = ThinPlateSplines(a, b)
    tps_two = ThinPlateSplines(b, a)
    chain = tps_one.compose_before(tps_two)
    assert(isinstance(chain, TransformChain))
    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain.apply(points)
    manual_res = tps_two.apply(tps_one.apply(points))
    assert (np.all(chain_res.points == manual_res.points))


def chain_tps_after_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps_one = ThinPlateSplines(a, b)
    tps_two = ThinPlateSplines(b, a)
    chain = tps_one.compose_after(tps_two)
    assert(isinstance(chain, TransformChain))
    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain.apply(points)
    manual_res = tps_one.apply(tps_two.apply(points))
    assert (np.all(chain_res.points == manual_res.points))


def chain_pwa_before_tps_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffine(a_tm, b)
    tps = ThinPlateSplines(b, a_tm)
    chain = pwa.compose_before(tps)
    assert(isinstance(chain, TransformChain))


def chain_pwa_after_tps_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffine(a_tm, b)
    tps = ThinPlateSplines(b, a_tm)
    chain = pwa.compose_after(tps)
    assert(isinstance(chain, TransformChain))


def chain_tps_before_pwa_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffine(a_tm, b)
    tps = ThinPlateSplines(b, a_tm)
    chain = tps.compose_before(pwa)
    assert(isinstance(chain, TransformChain))


def chain_tps_after_pwa_test():
    a_tm = TriMesh(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    pwa = PiecewiseAffine(a_tm, b)
    tps = ThinPlateSplines(b, a_tm)
    chain = tps.compose_after(pwa)
    assert(isinstance(chain, TransformChain))


def compose_tps_after_translation_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    t = Translation([3, 4])
    tps = ThinPlateSplines(a, b)
    chain = tps.compose_after(t)
    assert(isinstance(chain, TransformChain))


def manual_no_op_chain_test():
    points = PointCloud(np.random.random([10, 2]))
    t = Translation([3, 4])
    chain = TransformChain([t, t.pseudoinverse()])
    points_applied = chain.apply(points)
    assert(np.allclose(points_applied.points, points.points))


def chain_compose_before_tps_test():
    a = PointCloud(np.random.random([10, 2]))
    b = PointCloud(np.random.random([10, 2]))
    tps = ThinPlateSplines(a, b)

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
    tps = ThinPlateSplines(a, b)

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
    tps = ThinPlateSplines(a, b)

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
    tps = ThinPlateSplines(a, b)

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
    chain_2 = TransformChain([s.pseudoinverse(), t.pseudoinverse()])
    chain_1.compose_before_inplace(chain_2)

    points = PointCloud(np.random.random([10, 2]))
    chain_res = chain_1.apply(points)
    assert(np.allclose(points.points, chain_res.points))
