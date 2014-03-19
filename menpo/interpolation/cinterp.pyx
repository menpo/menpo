# distutils: language = c++
# distutils: sources = menpo/interpolation/cpp/interp2.cpp

import numpy as np
cimport numpy as np
import cython
cimport cython
from libcpp.string cimport string

# externally declare the exact NDARRAY struct and interpolate function
cdef extern from "cpp/interp2.h":
    cdef cppclass NDARRAY:
        NDARRAY(size_t, size_t*, size_t, np.float64_t*)
        const size_t n_dims
        const size_t *dims
        const size_t n_elems
        np.float64_t *data

    cdef void interpolate(const NDARRAY *F, const NDARRAY *X, const NDARRAY *Y,
                          const string type, double *out_data) except +

@cython.boundscheck(False)
def interp2(img not None, axis0_indices not None, axis1_indices not None,
            mode='bilinear'):
    """
    Given a multi-channel image and a set of sub-pixel coordinates, return the
    interpolated pixel values using the given ``mode``.

    Parameters
    ----------
    F : (N, 2, C) c-contiguous double ndarray
        Two dimensional input image, with ``C`` optional channels
    axis0_indices : (M,) ndarray
        The set of indices over the first axis.
    axis1_indices : (M,) ndarray
        The set of indices over the second axis.
    mode : {'bilinear', 'bicubic', 'nearest'}, optional
        The type of interpolation to be performed.

        Default: bilinear

    Returns
    -------
    interpolated : (M, C) c-contiguous double ndarray
        An array where each column is the interpolated values for the given
        channel.
    """

    # Convert Python input (indices), which could be integers, to doubles
    cdef np.ndarray[np.float64_t, ndim=3, mode='c'] F = np.array(
        img, dtype=np.float64, order='C')
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] axis0 = np.array(
            axis0_indices, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, mode='c'] axis1 = np.array(
            axis1_indices, dtype=np.float64)

    # Create the structs to pass in to the interpolate method. Have to
    # heap allocate the structs because Cython requires a default
    # constructor for stack allocated classes (which we can't have because
    # the NDARRAY struct has const members)
    cdef NDARRAY *F_array = new NDARRAY(3, <size_t*> F.shape, F.size,
                                        &F[0, 0, 0])
    cdef NDARRAY *axis0_array = new NDARRAY(1, <size_t*> axis0.shape,
                                            axis0.size, &axis0[0])
    cdef NDARRAY *axis1_array = new NDARRAY(1, <size_t*> axis1.shape,
                                            axis1.size, &axis1[0])

    # Allocate the output memory
    cdef np.ndarray[np.float64_t, ndim=2, mode='c'] out = np.zeros(
        [axis0.shape[0], F.shape[2]], dtype=np.float64)

    # Call the c-based interpolate method
    interpolate(F_array, axis0_array, axis1_array,
                <string> mode.encode('utf-8'), &out[0, 0])

    # Cleanup memory
    del F_array
    del axis0_array
    del axis1_array

    return out
