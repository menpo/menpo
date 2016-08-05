import numpy as np
import cython
cimport numpy as np
cimport cython
from libc.math cimport sqrt
from menpo.cy_utils cimport dtype_from_memoryview


ctypedef fused floats:
    np.float32_t
    np.float64_t

ctypedef fused integrals:
    np.int16_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int32_t
    np.int64_t


cdef normalize(floats[:, :] vec):
    """
    Normalize the given array of vectors inplace.

    Parameters
    ----------
    vec : (N, 3) c-contiguous float32/float64 ndarray
        The array of vectors to normalize
    """
    # Avoid divisions by almost 0 numbers
    # np.spacing(1) is equivalent to Matlab's eps
    cdef:
        floats mag
        Py_ssize_t n_elems = vec.shape[0], i = 0
        floats eps = np.spacing(1)

    for i in range(n_elems):
        mag = sqrt(vec[i, 0] * vec[i, 0] +
                   vec[i, 1] * vec[i, 1] +
                   vec[i, 2] * vec[i, 2])
        # Zero magnitude (or close to) causes massive values, so just set vector
        # to zero.
        if mag < eps:
            vec[i, 0] = 0.0
            vec[i, 1] = 0.0
            vec[i, 2] = 0.0
        else:
            vec[i, 0] /= mag
            vec[i, 1] /= mag
            vec[i, 2] /= mag


cdef inline floats[:, :] triangle_cross(floats[:, :] vertex,
                                        integrals[:, :] face):
    """
    The N x 3 cross product of the two sides of the triangles defined
    by the face array.

    Parameters
    ----------
    x : (N, 3) float32/float64 memory view
        First array to perform cross product with.
    y : (N, 3) float32/float64 memory view
        Second array to perform cross product with.

    Returns
    -------
    cross : (N, 3) c-contiguous float32/float64 ndarray
        The array of vectors representing the cross product between each
        corresponding vector.
    """
    vertex_dtype = dtype_from_memoryview(vertex)
    cdef:
        Py_ssize_t n_vert = vertex.shape[0], n_face = face.shape[0]
        Py_ssize_t i = 0, j = 0
        floats[:, :] z = np.zeros([n_face, 3], dtype=vertex_dtype)
        floats *v0
        floats *v1
        floats *v2
        floats x[3]
        floats y[3]

    for i in range(n_face):
        # This is dangerous since we just have a raw pointer, but since we know
        # our vectors are always [n, 3] we hard coded the 3 loop below.
        v0 = &vertex[face[i, 0], :][0]
        v1 = &vertex[face[i, 1], :][0]
        v2 = &vertex[face[i, 2], :][0]
        # Calculate triangle edge vectors
        for j in range(3):
            x[j] = v1[j] - v0[j]
            y[j] = v2[j] - v0[j]
        # Cross Product
        z[i, 0] = x[1] * y[2] - x[2] * y[1]
        z[i, 1] = x[2] * y[0] - x[0] * y[2]
        z[i, 2] = x[0] * y[1] - x[1] * y[0]

    return z


cpdef compute_vertex_normals(floats[:, :] vertex, integrals[:, :] face):
    """
    Compute the per-vertex normals of the vertices given a list of
    faces.

    Parameters
    ----------
    vertex : (N, 3) c-contiguous float32/float64 ndarray
        The list of points to compute normals for.
    face : (M, 3) c-contiguous int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    vertex_normal : (N, 3) c-contiguous float32/float64 ndarray
        The normal per vertex.
    :return:
    """
    cdef:
        Py_ssize_t i = 0, j = 0
        Py_ssize_t n_vert = vertex.shape[0]
        Py_ssize_t n_face = face.shape[0]
        floats[:, :] face_normal
        floats[:, :] vertex_normal = np.zeros_like(vertex)
        integrals f0, f1, f2

    # Calculate the cross product (per-face normal)
    face_normal = triangle_cross(vertex, face)
    normalize(face_normal)

    # Calculate per-vertex normal
    for i in range(n_face):
        f0 = face[i, 0]
        f1 = face[i, 1]
        f2 = face[i, 2]
        for j in range(3):
            vertex_normal[f0, j] += face_normal[i, j]
            vertex_normal[f1, j] += face_normal[i, j]
            vertex_normal[f2, j] += face_normal[i, j]
    # Normalize
    normalize(vertex_normal)

    return np.asarray(vertex_normal)


cpdef compute_face_normals(floats[:, :] vertex, integrals[:, :] face):
    """
    Compute per-face normals of the vertices given a list of
    faces.

    Parameters
    ----------
    vertex : (N, 3) c-contiguous float32/float64 ndarray
        The list of points to compute normals for.
    face : (M, 3) c-contiguous int16/int32/int64 ndarray
        The list of faces (triangle list).

    Returns
    -------
    face_normal : (M, 3) c-contiguous float32/float64 ndarray
        The normal per face.
    :return:
    """
    # Calculate the cross product (per-face normal)
    cdef floats[:, :] face_normal = triangle_cross(vertex, face)
    normalize(face_normal)

    return np.asarray(face_normal)
