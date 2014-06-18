#include <iostream>
#include <stdio.h>
#include <Python.h>

#include "check_cuda_errors.hpp"

bool cudaAssert(const cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        char buffer[512];
        sprintf(buffer, "Cuda error in file '%s' in line %i : %s.\n", file, line, cudaGetErrorString(err));
        throwRuntimeError(buffer);
        return false;
    }
    return true;
}

void throwRuntimeError(const char *what) {
    PyErr_SetString(PyExc_RuntimeError, what);
}
