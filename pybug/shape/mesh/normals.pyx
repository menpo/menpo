import numpy as np
import cython
cimport numpy as np
cimport cython

ctypedef np.float64_t FLOAT64_T
ctypedef fused integrals:
    np.uint32_t
    np.uint64_t
    np.int32_t
    np.int64_t

cdef np.ndarray[FLOAT64_T, ndim=2] normalise(np.ndarray[FLOAT64_T, ndim=2] vec):
    """
    Normalise the given vector
    :param vec: N x 3 vector
    :return: Normalised N x 3 vector
    """
    # Avoid divisions by almost 0 numbers
    # np.spacing(1) is equivalent to Matlab's eps
    cdef np.ndarray[FLOAT64_T, ndim=1] d = np.sqrt(np.sum(vec ** 2, axis=1))
    d[d < np.spacing(1)] = 1.0
    return vec / d[..., None]


cdef inline np.ndarray[FLOAT64_T, ndim=2] cross(double[:, :] x,
                                                double[:, :] y):
    """
    The N x 3 cross product (returns the vectors orthogonal to x and y)
    :param x: N x 3 vector
    :param y: N x 3 vector
    :return: The N x 3 vector representing the cross product
    """
    cdef np.ndarray[FLOAT64_T, ndim=2] z = np.empty_like(x)
    cdef int n = x.shape[0]
    for i in range(n):
        z[i, 0] = x[i, 1] * y[i, 2] - x[i, 2] * y[i, 1]
        z[i, 1] = x[i, 2] * y[i, 0] - x[i, 0] * y[i, 2]
        z[i, 2] = x[i, 0] * y[i, 1] - x[i, 1] * y[i, 0]

    return z


cpdef compute_normals(np.ndarray[FLOAT64_T, ndim=2] vertex,
                      np.ndarray[integrals, ndim=2] face):
    """
    Compute the per-vertex and per-face normal of the vertices given a list of
    faces.
    :param vertex: The list of points to compute normals for
    :type vertex: ndarray [N, 3]
    :param face: The list of faces (triangle list)
    :type face: ndarray [M, 3]
    :return:
    """
    cdef int nface = face.shape[0]
    cdef int nvert = vertex.shape[0]

    # Calculate the cross product (per-face normal)
    cdef np.ndarray[FLOAT64_T, ndim=2] face_normal = cross(
        vertex[face[:, 1], :] - vertex[face[:, 0], :],
        vertex[face[:, 2], :] - vertex[face[:, 0], :])
    face_normal = normalise(face_normal)

    # Calculate per-vertex normal
    cdef np.ndarray[FLOAT64_T, ndim=2] vertex_normal = np.zeros([nvert, 3])
    cdef int f0, f1, f2
    for i in range(nface):
        f0 = face[i, 0]
        f1 = face[i, 1]
        f2 = face[i, 2]
        for j in range(3):
            vertex_normal[f0, j] += face_normal[i, j]
            vertex_normal[f1, j] += face_normal[i, j]
            vertex_normal[f2, j] += face_normal[i, j]

    # Normalize
    vertex_normal = normalise(vertex_normal)

    # Enforce that the normals are outward
    cdef np.ndarray[FLOAT64_T, ndim=2] v = vertex - np.mean(vertex)[..., None]
    cdef np.ndarray[FLOAT64_T, ndim=1] s = np.sum(v * vertex_normal, axis=1)
    if np.sum(np.greater(s, 0)) < np.sum(np.less(s, 0)):
        # Flip
        vertex_normal = -vertex_normal
        face_normal = -face_normal

    return vertex_normal, face_normal