"""A TriMesh checks its triangle list when it is made (unless skip_checks)."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from menpo.image import Image
from menpo.shape import ColouredTriMesh, TexturedTriMesh, TriMesh

POINTS = np.random.RandomState(0).rand(5, 3)
GOOD = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])


def make(trilist, **kwargs):
    return TriMesh(POINTS, trilist, **kwargs)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.uint32, np.uint64])
def test_integer_trilists_are_kept_as_they_are(dtype):
    mesh = make(GOOD.astype(dtype))
    assert mesh.trilist.dtype == dtype
    assert_array_equal(mesh.trilist, GOOD)


def test_a_list_of_lists_is_accepted():
    assert_array_equal(make(GOOD.tolist()).trilist, GOOD)


def test_a_float_trilist_of_whole_numbers_becomes_integers():
    mesh = make(GOOD.astype(np.float64))
    assert mesh.trilist.dtype.kind == "i"
    assert_array_equal(mesh.trilist, GOOD)
    # and the things that need to index with it work
    assert mesh.tri_areas().shape == (3,)
    assert mesh.vertex_normals().shape == (5, 3)


def test_float32_and_a_whole_number_written_with_a_fraction_of_zero_are_fine():
    assert_array_equal(make(GOOD.astype(np.float32)).trilist, GOOD)
    assert_array_equal(make(GOOD + 0.0).trilist, GOOD)


@pytest.mark.parametrize(
    "trilist, message",
    [
        (np.array([[0, 1, 2.5]]), "whole numbers.*2.5"),
        (np.array([[0, 1, np.nan]]), "whole numbers"),
        (np.array([[0, 1, np.inf]]), "whole numbers"),
        (np.array([[True, False, True]]), "integer point indices.*bool"),
        (np.array([["0", "1", "2"]]), "integer point indices"),
        (np.array([0, 1, 2]), r"shape \(n_tris, 3\).*\(3,\)"),
        (np.array([[0, 1, 2, 3]]), r"shape \(n_tris, 3\).*\(1, 4\)"),
        (np.zeros((2, 3, 1), dtype=int), r"shape \(n_tris, 3\)"),
        (np.array([[0, 1, -1]]), "negative point index -1"),
        (np.array([[0, 1, 5]]), "refers to point 5.*only 5 points.*0 to 4"),
        (np.array([[0, 1, 200]]), "refers to point 200"),
    ],
)
def test_a_bad_trilist_is_an_error_that_says_what_is_wrong(trilist, message):
    with pytest.raises(ValueError, match=message):
        make(trilist)


def test_no_triangles_is_fine():
    mesh = make(np.zeros((0, 3), dtype=int))
    assert mesh.n_tris == 0


def test_no_trilist_is_a_delaunay_triangulation():
    mesh = TriMesh(np.random.RandomState(1).rand(20, 2))
    assert mesh.trilist.shape[1] == 3 and mesh.trilist.max() < 20


def test_skip_checks_lets_anything_through_untouched():
    floats = GOOD.astype(np.float64)
    assert make(floats, skip_checks=True).trilist.dtype == np.float64
    for bad in [
        np.array([[0, 1, 9]]),
        np.array([[0, 1, -1]]),
        np.array([[0, 1, 2, 3]]),
    ]:
        assert_array_equal(make(bad, skip_checks=True).trilist, bad)


def test_copy_is_still_honoured_when_the_trilist_is_valid():
    trilist = GOOD.astype(np.int64)
    assert not np.shares_memory(make(trilist).trilist, trilist)
    assert np.shares_memory(make(trilist, copy=False).trilist, trilist)
    assert np.shares_memory(
        make(trilist, copy=False, skip_checks=True).trilist, trilist
    )


def test_the_check_uses_the_number_of_points_of_this_mesh():
    trilist = np.array([[0, 1, 4]])
    TriMesh(POINTS, trilist)  # 5 points: fine
    with pytest.raises(ValueError, match="only 4 points"):
        TriMesh(POINTS[:4], trilist)


def test_coloured_and_textured_meshes_check_the_trilist_too():
    colours = np.ones((5, 3))
    texture = Image(np.random.RandomState(2).rand(3, 4, 4))
    tcoords = np.random.RandomState(3).rand(5, 2)
    bad = np.array([[0, 1, 9]])
    with pytest.raises(ValueError, match="refers to point 9"):
        ColouredTriMesh(POINTS, bad, colours)
    with pytest.raises(ValueError, match="refers to point 9"):
        TexturedTriMesh(POINTS, tcoords, texture, bad)
    assert ColouredTriMesh(POINTS, bad, colours, skip_checks=True).n_tris == 1
    assert TexturedTriMesh(POINTS, tcoords, texture, bad, skip_checks=True).n_tris == 1
    fine = ColouredTriMesh(POINTS, GOOD.astype(float), colours)
    assert fine.trilist.dtype.kind == "i"
    fine = TexturedTriMesh(POINTS, tcoords, texture, GOOD.astype(float))
    assert fine.trilist.dtype.kind == "i"


def test_meshes_made_inside_menpo_from_a_known_good_trilist_still_work():
    grid = TriMesh.init_2d_grid((4, 5))
    assert grid.trilist.max() == 19
    coloured = ColouredTriMesh.init_2d_grid((4, 5))
    textured = TexturedTriMesh.init_2d_grid(
        (4, 5), texture=Image(np.random.rand(3, 4, 5))
    )
    assert coloured.n_tris == textured.n_tris == grid.n_tris
    vector = textured.as_vector()
    assert textured.from_vector(vector).n_tris == grid.n_tris
