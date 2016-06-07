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

cdef np.ndarray[FLOAT64_T, ndim=2] normalize(np.ndarray[FLOAT64_T, ndim=2] vec):
    """
    Normalize the given array of vectors.

    Parameters
    ----------
    vec : (N, 3) c-contiguous double ndarray
        The array of vectors to normalize

    Returns
    -------
    normalized : (N, 3) c-contiguous double ndarray
        Normalized array of vectors.
    """
    # Avoid divisions by almost 0 numbers
    # np.spacing(1) is equivalent to Matlab's eps
    cdef np.ndarray[FLOAT64_T, ndim=1] d = np.sqrt(np.sum(vec ** 2, axis=1))
    d[d < np.spacing(1)] = 1.0
    return vec / d[..., None]


cdef inline np.ndarray[FLOAT64_T, ndim=2] cross(double[:, :] x,
                                                double[:, :] y):
    """
    The N x 3 cross product (returns the vectors orthogonal
    to ``x`` and ``y``). This performs the cross product on each (3, 1) vector
    in the two arrays. Assumes ``x`` and ``y`` have the same shape.

    Parameters
    ----------
    x : (N, 3) double memory view
        First array to perform cross product with.
    y : (N, 3) double memory view
        Second array to perform cross product with.

    Returns
    -------
    cross : (N, 3) c-contiguous double ndarray
        The array of vectors representing the cross product between each
        corresponding vector.
    """
    cdef:
        np.ndarray[FLOAT64_T, ndim=2] z = np.empty_like(x)
        Py_ssize_t n = x.shape[0], i = 0
    for i in range(n):
        z[i, 0] = x[i, 1] * y[i, 2] - x[i, 2] * y[i, 1]
        z[i, 1] = x[i, 2] * y[i, 0] - x[i, 0] * y[i, 2]
        z[i, 2] = x[i, 0] * y[i, 1] - x[i, 1] * y[i, 0]

    return z


cpdef compute_normals(np.ndarray[FLOAT64_T, ndim=2] vertex,
                      np.ndarray[integrals, ndim=2] face):
    """
    Compute the per-vertex and per-face normal of the vertices given a list of
    faces. Ensures that all the normals are pointing in a consistent direction
    (to avoid 'inverted' normals).

    Parameters
    ----------
    vertex : (N, 3) c-contiguous double ndarray
        The list of points to compute normals for.
    face : (M, 3) c-contiguous double ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) c-contiguous double ndarray
        The normal per vertex.
    face_normal : (M, 3) c-contiguous double ndarray
        The normal per face.
    :return:
    """
    cdef int nface = face.shape[0]
    cdef int nvert = vertex.shape[0]

    # Calculate the cross product (per-face normal)
    cdef np.ndarray[FLOAT64_T, ndim=2] face_normal = cross(
        vertex[face[:, 1], :] - vertex[face[:, 0], :],
        vertex[face[:, 2], :] - vertex[face[:, 0], :])
    face_normal = normalize(face_normal)

    # Calculate per-vertex normal
    cdef np.ndarray[FLOAT64_T, ndim=2] vertex_normal = np.zeros([nvert, 3])
    cdef integrals f0, f1, f2
    for i in range(nface):
        f0 = face[i, 0]
        f1 = face[i, 1]
        f2 = face[i, 2]
        for j in range(3):
            vertex_normal[f0, j] += face_normal[i, j]
            vertex_normal[f1, j] += face_normal[i, j]
            vertex_normal[f2, j] += face_normal[i, j]

    # Normalize
    vertex_normal = normalize(vertex_normal)

    return vertex_normal, face_normal
