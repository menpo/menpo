import numpy as np
from pybug.shape import TriMesh


def test_trimesh_creation():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    TriMesh(points, trilist)


def test_trimesh_n_dims():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist)
    assert(trimesh.n_dims == 3)


def test_trimesh_n_points():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist)
    assert(trimesh.n_points == 4)


def test_trimesh_n_tris():
    points = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    trimesh = TriMesh(points, trilist)
    assert(trimesh.n_tris == 2)
