"""index_alpha_beta finds each point's triangle with a KD-tree. It must give exactly what testing every point against
every triangle gave (that is `alpha_beta` and `containment_from_alpha_beta`, kept as the reference here)."""

import tracemalloc

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.spatial import Delaunay

from menpo.shape import TriMesh
from menpo.transform import PiecewiseAffine
from menpo.transform.piecewiseaffine import base as pwa_base
from menpo.transform.piecewiseaffine.base import (
    TriangleContainmentError,
    alpha_beta,
    barycentric_vectors,
    containment_from_alpha_beta,
    index_alpha_beta,
)


def reference(i, ij, ik, points):
    """The original: every point against every triangle, a matrix of n_points x n_tris."""
    alpha, beta = alpha_beta(i, ij, ik, points)
    index = containment_from_alpha_beta(alpha, beta)
    rows = np.arange(len(points))
    return index, alpha[rows, index], beta[rows, index]


def assert_same(got, expected):
    """The same triangle for every point, and the same alpha and beta (to a tolerance that is 1e-12 of them: the two
    ways of computing them sum the same products, but a platform that fuses multiply-adds may round the last bit
    differently)."""
    assert_array_equal(got[0], expected[0])
    np.testing.assert_allclose(got[1], expected[1], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(got[2], expected[2], rtol=1e-12, atol=1e-14)


def delaunay_mesh(n, extent=300.0, seed=0):
    rng = np.random.RandomState(seed)
    corners = [[0, 0], [0, extent], [extent, 0], [extent, extent]]
    points = np.vstack([rng.rand(n, 2) * extent, corners])
    return points, Delaunay(points).simplices


def inside_points(n, extent=300.0, seed=1):
    return np.random.RandomState(seed).rand(n, 2) * (extent - 2) + 1


@pytest.mark.parametrize("n_landmarks", [3, 68, 400])
@pytest.mark.parametrize("extent", [1.0, 300.0, 1e4])
@pytest.mark.parametrize("seed", [0, 1])
def test_the_same_triangle_alpha_and_beta_as_testing_every_triangle(
    n_landmarks, extent, seed
):
    points, trilist = delaunay_mesh(n_landmarks, extent, seed)
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = inside_points(3000, extent, seed + 10)
    expected = reference(i, ij, ik, queries)
    got = index_alpha_beta(i, ij, ik, queries)
    assert_same(got, expected)
    assert got[0].dtype == np.uint32 and got[1].dtype == got[2].dtype == np.float64
    assert got[0].shape == got[1].shape == got[2].shape == (3000,)


def test_points_just_inside_every_vertex_and_edge_are_found_in_that_triangle():
    # (a point exactly on an edge can be "outside" by a rounding error of 1e-17 in both the old and the new
    # arithmetic, so what is tested is the points a hair inside: the corners of the bounding balls)
    points, trilist = delaunay_mesh(60)
    i, ij, ik = barycentric_vectors(points, trilist)
    corners = points[trilist]  # (n_tris, 3, 2)
    centroids = corners.mean(axis=1)
    midpoints = (corners + np.roll(corners, -1, axis=1)) / 2
    targets = np.concatenate([corners, midpoints], axis=1)  # (n_tris, 6, 2)
    queries = (targets + 1e-6 * (centroids[:, None, :] - targets)).reshape(-1, 2)
    expected_tri = np.repeat(np.arange(len(trilist)), 6)
    got = index_alpha_beta(i, ij, ik, queries)
    assert_array_equal(got[0], expected_tri)
    assert_same(got, reference(i, ij, ik, queries))


def test_the_centroid_of_every_triangle_is_in_that_triangle():
    points, trilist = delaunay_mesh(200)
    i, ij, ik = barycentric_vectors(points, trilist)
    centroids = points[trilist].mean(axis=1)
    tri, alpha, beta = index_alpha_beta(i, ij, ik, centroids)
    assert_array_equal(tri, np.arange(len(trilist)))
    np.testing.assert_allclose(alpha, 1 / 3)
    np.testing.assert_allclose(beta, 1 / 3)


@pytest.mark.parametrize("chunk", [1024, 2, 1])
def test_a_point_in_overlapping_triangles_belongs_to_the_last_one(chunk, monkeypatch):
    monkeypatch.setattr(
        pwa_base, "_TRIANGLES_PER_CHUNK", chunk
    )  # the last one also across chunks
    # three copies of a triangle and a bigger one that covers it, in different orders
    points = np.array([[0, 0], [10, 0], [0, 10], [20, 0], [0, 20.0]])
    trilist = np.array([[0, 1, 2], [0, 3, 4], [0, 1, 2], [0, 3, 4], [0, 1, 2]])
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = np.random.RandomState(2).rand(500, 2) * 4 + 1  # inside all of them
    expected = reference(i, ij, ik, queries)
    got = index_alpha_beta(i, ij, ik, queries)
    assert (got[0] == 4).all()
    assert_same(got, expected)


def test_points_outside_the_mesh_raise_with_the_same_mask_as_before():
    points, trilist = delaunay_mesh(40)
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = np.vstack(
        [inside_points(50), [[-5, 10], [10, 400], [301, 301], [150, 150]]]
    )
    with pytest.raises(TriangleContainmentError) as expected:
        reference(i, ij, ik, queries)
    with pytest.raises(TriangleContainmentError) as got:
        index_alpha_beta(i, ij, ik, queries)
    assert_array_equal(
        got.value.points_outside_source_domain,
        expected.value.points_outside_source_domain,
    )
    assert got.value.points_outside_source_domain.sum() == 3


def test_points_that_are_not_finite_are_outside_the_mesh():
    points, trilist = delaunay_mesh(40)
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = inside_points(20)
    queries[3] = [np.nan, 5.0]
    queries[7] = [np.inf, 5.0]
    queries[11] = [5.0, -np.inf]
    with np.errstate(all="ignore"):
        with pytest.raises(TriangleContainmentError) as expected:
            reference(i, ij, ik, queries)
    with pytest.raises(TriangleContainmentError) as got:
        index_alpha_beta(i, ij, ik, queries)
    assert_array_equal(
        got.value.points_outside_source_domain,
        expected.value.points_outside_source_domain,
    )
    assert got.value.points_outside_source_domain.nonzero()[0].tolist() == [3, 7, 11]


def test_no_points_gives_empty_results():
    points, trilist = delaunay_mesh(10)
    i, ij, ik = barycentric_vectors(points, trilist)
    tri, alpha, beta = index_alpha_beta(i, ij, ik, np.zeros((0, 2)))
    assert tri.shape == alpha.shape == beta.shape == (0,)
    assert tri.dtype == np.uint32


def test_a_triangle_with_no_area_does_not_steal_points():
    points, trilist = delaunay_mesh(30)
    # a flat triangle first (it is triangle 0), so that a real triangle wins any tie by its higher index
    points = np.vstack([points, [[150.0, 150.0]]])
    trilist = np.vstack([[[0, 0, 0]], trilist])
    i, ij, ik = barycentric_vectors(points, trilist)
    centroids = points[trilist[1:]].mean(axis=1)
    with np.errstate(all="ignore"):
        tri, alpha, beta = index_alpha_beta(i, ij, ik, centroids)
    assert_array_equal(tri, np.arange(1, len(trilist)))


def test_looking_in_chunks_of_triangles_changes_nothing(monkeypatch):
    points, trilist = delaunay_mesh(300, seed=3)
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = inside_points(5000, seed=4)
    expected = reference(i, ij, ik, queries)
    monkeypatch.setattr(pwa_base, "_TRIANGLES_PER_CHUNK", 7)
    got = index_alpha_beta(i, ij, ik, queries)
    assert_same(got, expected)


def test_a_dense_mesh_does_not_need_a_matrix_of_points_by_triangles():
    # 4000 triangles and 30000 points: the matrix used to take about 7 GB
    points, trilist = delaunay_mesh(2000)
    i, ij, ik = barycentric_vectors(points, trilist)
    queries = inside_points(30000)
    tracemalloc.start()
    try:
        index_alpha_beta(i, ij, ik, queries)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 300e6


def test_a_piecewise_affine_warp_of_a_dense_mesh_with_the_same_source_and_target_is_the_identity():
    points, trilist = delaunay_mesh(1500)
    mesh = TriMesh(points, trilist)
    warp = PiecewiseAffine(mesh, TriMesh(points.copy(), trilist))
    queries = inside_points(20000)
    np.testing.assert_allclose(warp.apply(queries), queries, atol=1e-9)


def test_a_piecewise_affine_warp_moves_the_points_with_their_triangle():
    points, trilist = delaunay_mesh(100)
    shift = np.array([3.0, -2.0])
    warp = PiecewiseAffine(TriMesh(points, trilist), TriMesh(points + shift, trilist))
    queries = inside_points(2000)
    np.testing.assert_allclose(warp.apply(queries), queries + shift, atol=1e-9)
