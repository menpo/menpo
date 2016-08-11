import numpy as np
cimport numpy as np
cimport cython


ctypedef fused DOUBLE_TYPES:
    float
    double


cdef extern from "cpp/central_difference.h":
    void central_difference[T](const T* input, const Py_ssize_t rows,
                               const Py_ssize_t cols, const Py_ssize_t n_channels,
                               T* output)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gradient_cython(np.ndarray[DOUBLE_TYPES, ndim=3] input):

    cdef Py_ssize_t n_channels = input.shape[0]
    cdef Py_ssize_t rows = input.shape[1]
    cdef Py_ssize_t cols = input.shape[2]
    # Maintain the dtype that was passed in (float or double)
    dtype = input.dtype
    cdef np.ndarray[DOUBLE_TYPES, ndim=3] output = np.zeros((n_channels * 2,
                                                            rows, cols),
                                                            dtype=dtype)

    central_difference(&input[0,0,0], rows, cols, n_channels,
                       &output[0,0,0])

    return output
