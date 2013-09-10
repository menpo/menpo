# distutils: language = c++
# distutils: sources = pybug/features/cpp/ImageWindowIterator.cpp

import numpy as np
cimport numpy as np
import cython
cimport cython
from libcpp.string cimport string


cdef extern from "math.h":
    double ceil(double)
    double round(double)
    double floor(double)

cdef extern from "cpp/ImageWindowIterator.h":
    cdef cppclass ImageWindowIterator:
        ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth,
			unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
			unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale, WindowFeature *windowFeature)
        void apply(double *outputImage, int *windowsCenters)
	    void print_information()