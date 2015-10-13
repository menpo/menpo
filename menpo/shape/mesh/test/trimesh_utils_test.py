from menpo.shape import TriMesh
import numpy as np


def utils_mesh():
    trilist = np.array([[0, 1, 2],
                        [0, 2, 3]])
    points = np.array([[0, 0],
                       [1, 0],
                       [1, 1],
                       [0, 1]])
    return TriMesh(points, trilist=trilist)


gt_edge_indices = np.array([[0, 1],
                            [0, 2],
                            [1, 2],
                            [2, 3],
                            [2, 0],
                            [3, 0]])

gt_edge_vectors = np.array([[1, 0],
                            [1, 1],
                            [0, 1],
                            [-1, 0],
                            [1, 1],
                            [0, 1]])

gt_unique_edge_vectors = np.array([[1, 0],
                                   [1, 1],
                                   [0, 1],
                                   [0, 1],
                                   [-1, 0]])

gt_edge_lengths = np.array([1., 1.41421356, 1., 1., 1.41421356, 1.])

gt_unique_edge_lengths = np.array([1., 1.41421356, 1., 1., 1.])

gt_tri_areas = np.array([0.5, 0.5])


def test_edge_indices():
    assert np.all(utils_mesh().edge_indices() == gt_edge_indices)


def test_edge_vectors():
    assert np.all(utils_mesh().edge_vectors() == gt_edge_vectors)


def test_unique_edge_vectors():
    assert np.all(utils_mesh().unique_edge_vectors() == gt_unique_edge_vectors)


def test_edge_lengths():
    assert np.allclose(utils_mesh().edge_lengths(), gt_edge_lengths)


def test_unique_edge_lengths():
    assert np.allclose(utils_mesh().unique_edge_lengths(),
                       gt_unique_edge_lengths)


def test_tri_areas():
    assert np.allclose(utils_mesh().tri_areas(), gt_tri_areas)


def test_2d_trimesh_2d_positive_areas():
    t = TriMesh(np.array([[0, 0], [0, 1],
                          [1, 1], [1, 0]], dtype=np.float),
                trilist=np.array([[0, 2, 3], [0, 2, 1]]))
    assert np.all(t.tri_areas() > 0)


def test_mean_tri_area():
    assert utils_mesh().mean_tri_area() == 0.5


def test_mean_edge_length():
    assert np.allclose(utils_mesh().mean_edge_length(),
                       np.mean(gt_unique_edge_lengths))


def test_mean_edge_length_not_unique():
    assert np.allclose(utils_mesh().mean_edge_length(unique=False),
                       np.mean(gt_edge_lengths))
