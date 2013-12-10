import numpy as np
from numpy.testing import assert_allclose
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


def test_trimesh_face_normals():
    points = np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    expected_normals = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3, np.sqrt(3)/3],
                                 [-0, -0, 1]])
    trimesh = TriMesh(points, trilist)
    face_normals = trimesh.face_normals
    assert_allclose(face_normals, expected_normals)


def test_trimesh_vertex_normals():
    points = np.array([[0.0, 0.0, -1.0],
                       [1.0, 0.0, 0.0],
                       [1.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    trilist = np.array([[0, 1, 3],
                        [1, 2, 3]])
    # 0 and 2 are the corner of the triangles and so the maintain the
    # face normals. The other two are the re-normalised vertices:
    # normalise(n0 + n2)
    expected_normals = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3, np.sqrt(3)/3],
                                 [-0.32505758,  -0.32505758, 0.88807383],
                                 [0, 0, 1],
                                 [-0.32505758,  -0.32505758, 0.88807383]])
    trimesh = TriMesh(points, trilist)
    vertex_normals = trimesh.vertex_normals
    assert_allclose(vertex_normals, expected_normals)