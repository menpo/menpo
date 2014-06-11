# distutils: language = c++
# distutils: sources = menpo/cuda/cu/cutools.cu

from libcpp cimport bool

cdef extern from "cu/cutools.hpp":
    bool is_cuda_available_()

def is_cuda_available():
    return is_cuda_available_()

