# distutils: language = c++
# distutils: sources = menpo/feature/cpp/central_difference.cpp

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "cpp/central_difference.h":
    void central_difference(const double* input, const int rows,
                            const int cols, const int n_channels,
                            double* output)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef gradient_cython(np.ndarray[double, ndim=3] input):

    cdef long long n_channels = input.shape[0]
    cdef long long rows = input.shape[1]
    cdef long long cols = input.shape[2]
    cdef np.ndarray[double, ndim=3] output = np.zeros((n_channels * 2,
                                                       rows, cols))

    central_difference(&input[0,0,0], rows, cols, n_channels,
                       &output[0,0,0])

    return output
