import numpy as np


def _normalize(v):
    return v / np.sqrt((v ** 2).sum(axis=1)[:, None])


def compute_face_normals(points, trilist):
    """
    Compute per-face normals of the vertices given a list of
    faces.

    Parameters
    ----------
    vertex : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    face : (M, 3) int16/int32/int64 ndarray
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
    vertex : (N, 3) float32/float64 ndarray
        The list of points to compute normals for.
    face : (M, 3) int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) float32/float64 ndarray
        The normal per vertex.
    :return:
    """
    tri_normals = compute_face_normals(points, trilist)

    tris_per_vertex = [[] for _ in points]

    for i, vertices in enumerate(trilist):
        for v in vertices:
            tris_per_vertex[v].append(i)

    return _normalize(np.array(
        [np.mean([tri_normals[t] for t in tris], axis=0)
         for tris in tris_per_vertex]))
