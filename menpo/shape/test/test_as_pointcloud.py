from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from menpo.image import Image
from menpo.shape import (
    ColouredTriMesh,
    LabelledPointUndirectedGraph,
    PointCloud,
    PointDirectedGraph,
    PointTree,
    PointUndirectedGraph,
    TexturedTriMesh,
    TriMesh,
)


def _shapes():
    rng = np.random.RandomState(0)
    points3 = rng.rand(6, 3)
    points2 = rng.rand(6, 2)
    tri = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5]])
    adjacency = csr_matrix(np.diag(np.ones(5), 1) + np.diag(np.ones(5), -1))
    # vertex 0 is the parent of 1 and 2, 1 of 3 and 4, 2 of 5
    tree = csr_matrix((np.ones(5), ([0, 0, 1, 1, 2], [1, 2, 3, 4, 5])), shape=(6, 6))
    masks = OrderedDict(
        [
            ("a", np.array([1, 1, 1, 0, 0, 0], bool)),
            ("b", np.array([0, 0, 0, 1, 1, 1], bool)),
        ]
    )
    return {
        "PointCloud": PointCloud(points3),
        "TriMesh": TriMesh(points3, tri),
        "TriMesh2D": TriMesh(points2, tri),
        "ColouredTriMesh": ColouredTriMesh(points3, tri, colours=rng.rand(6, 3)),
        "TexturedTriMesh": TexturedTriMesh(
            points3, rng.rand(6, 2), Image(rng.rand(3, 8, 8)), tri
        ),
        "PointUndirectedGraph": PointUndirectedGraph(points2, adjacency),
        "PointDirectedGraph": PointDirectedGraph(points2, adjacency),
        "PointTree": PointTree(points2, tree, root_vertex=0),
        "LabelledPointUndirectedGraph": LabelledPointUndirectedGraph(
            points2, adjacency, masks
        ),
    }


SHAPES = _shapes()


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_shape_becomes_a_plain_point_cloud_with_the_same_points(name):
    shape = SHAPES[name]
    cloud = shape.as_pointcloud()
    assert type(cloud) is PointCloud
    assert_array_equal(cloud.points, shape.points)
    assert cloud.n_dims == shape.n_dims and cloud.n_points == shape.n_points


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_default_is_a_copy_that_does_not_change_the_original(name):
    shape = SHAPES[name]
    before = shape.points.copy()
    cloud = shape.as_pointcloud()
    assert not np.shares_memory(cloud.points, shape.points)
    cloud.points[:] = 0
    assert_array_equal(shape.points, before)


@pytest.mark.parametrize("name", ["TriMesh", "PointUndirectedGraph", "PointCloud"])
def test_copy_false_shares_the_points(name):
    shape = SHAPES[name].copy()
    cloud = shape.as_pointcloud(copy=False)
    assert np.shares_memory(cloud.points, shape.points)


def test_landmarks_come_with_it_and_are_not_shared():
    mesh = SHAPES["TriMesh"].copy()
    mesh.landmarks["tips"] = PointCloud(mesh.points[:2])
    mesh.landmarks["more"] = PointCloud(mesh.points[2:5])
    cloud = mesh.as_pointcloud()
    assert set(cloud.landmarks.group_labels) == {"tips", "more"}
    assert_array_equal(cloud.landmarks["tips"].points, mesh.points[:2])
    cloud.landmarks["extra"] = PointCloud(mesh.points[:1])  # a new group
    cloud.landmarks["tips"].points[:] = 99  # and a change to one of them
    assert "extra" not in mesh.landmarks.group_labels
    assert_array_equal(mesh.landmarks["tips"].points, mesh.points[:2])


def test_a_shape_without_landmarks_gives_a_cloud_without_landmarks():
    assert SHAPES["PointUndirectedGraph"].as_pointcloud().landmarks.n_groups == 0


def test_the_path_comes_with_it_when_there_is_one():
    mesh = SHAPES["TriMesh"].copy()
    assert not hasattr(mesh.as_pointcloud(), "path")
    mesh.path = Path("a") / "mesh.obj"
    assert mesh.as_pointcloud().path == Path("a") / "mesh.obj"


def test_a_plain_point_cloud_gives_an_equal_independent_one():
    cloud = SHAPES["PointCloud"]
    other = cloud.as_pointcloud()
    assert other is not cloud and type(other) is PointCloud
    assert_array_equal(other.points, cloud.points)
