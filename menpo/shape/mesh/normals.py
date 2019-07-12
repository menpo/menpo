import numpy as np


def _normalize(v):
    return np.nan_to_num(v / np.sqrt((v ** 2).sum(axis=1, keepdims=True)))


def compute_face_normals(points, trilist):
    """
    Compute per-face normals of the vertices given a list of
    faces.

    Parameters
    ----------
    points : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    trilist : (M, 3) int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    face_normal : (M, 3) float32/float64 ndarray
        The normal per face.
    :return:
    """
    pt = points[trilist]
    a, b, c = pt[:, 0], pt[:, 1], pt[:, 2]
    norm = np.cross(b - a, c - a)
    return _normalize(norm)


def compute_vertex_normals(points, trilist):
    """
    Compute the per-vertex normals of the vertices given a list of
    faces.

    Parameters
    ----------
    points : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    trilist : (M, 3) int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) float32/float64 ndarray
        The normal per vertex.
    """
    face_normals = compute_face_normals(points, trilist)

    vertex_normals = np.zeros(points.shape, dtype=points.dtype)
    np.add.at(vertex_normals, trilist[:, 0], face_normals)
    np.add.at(vertex_normals, trilist[:, 1], face_normals)
    np.add.at(vertex_normals, trilist[:, 2], face_normals)

    return _normalize(vertex_normals)
