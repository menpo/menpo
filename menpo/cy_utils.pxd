cimport cython
cimport numpy as np


cdef inline np.dtype dtype_from_memoryview(cython.view.memoryview arr):
    return np.dtype(arr.view.format)
