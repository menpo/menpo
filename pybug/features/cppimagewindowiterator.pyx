# distutils: language = c++
# distutils: sources = pybug/features/cpp/ImageWindowIterator.cpp pybug/features/cpp/WindowFeature.cpp pybug/features/cpp/HOG.cpp

import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "math.h":
    double ceil(double)
    double round(double)
    double floor(double)

cdef extern from "cpp/ImageWindowIterator.h":
    cdef cppclass ImageWindowIterator:
        ImageWindowIterator(double *image, unsigned int imageHeight, unsigned int imageWidth,
			unsigned int windowHeight, unsigned int windowWidth, unsigned int windowStepHorizontal,
			unsigned int windowStepVertical, bool enablePadding, bool imageIsGrayscale)
        void apply(double *outputImage, int *windowsCenters, WindowFeature *windowFeature)
        void print_information()
        unsigned int _numberOfWindowsHorizontally, \
            _numberOfWindowsVertically, _numberOfWindows

cdef extern from "cpp/WindowFeature.h":
    cdef cppclass WindowFeature:
        void apply(double *windowImage, bool imageIsGrayscale,
                   double *descriptorVector)
        unsigned int descriptorLengthPerWindow


cdef extern from "cpp/HOG.h":
    cdef cppclass HOG(WindowFeature):
        HOG(unsigned int windowHeight, unsigned int windowWidth, unsigned int method, unsigned int numberOfOrientationBins, unsigned int cellHeightAndWidthInPixels,
			unsigned int blockHeightAndWidthInCells, bool enableSignedGradients, double l2normClipping)
        void apply(double *windowImage, bool imageIsGrayscale, double *descriptorVector)

cdef class CppImageWindowIterator:
    cdef ImageWindowIterator* iterator
    cdef unsigned int windowHeight, windowWidth

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] image,
                  unsigned int windowHeight, unsigned int windowWidth,
                  unsigned int windowStepHorizontal,
                  unsigned int windowStepVertical, bool enablePadding):
        height = image.shape[0]
        width = image.shape[1]
        imageIsGrayscale = image.shape[2] == 1
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] image_f = \
            np.require(image, requirements='F')
        self.iterator = new ImageWindowIterator(&image_f[0,0,0], height,
                                                width ,
                                                windowHeight,windowWidth,
                                                windowStepHorizontal,windowStepVertical,
                                                enablePadding,imageIsGrayscale)
        self.windowHeight = windowHeight
        self.windowWidth = windowWidth

    def __str__(self):
        return 'state goes here'

    def HOG(self, method, numberOfOrientationBins, cellHeightAndWidthInPixels,
            blockHeightAndWidthInCells, enableSignedGradients,
            l2normClipping):

        cdef HOG *hog = new HOG(self.windowHeight, self.windowWidth, method,
                                numberOfOrientationBins,
                                cellHeightAndWidthInPixels,
                                blockHeightAndWidthInCells,
                                enableSignedGradients, l2normClipping)
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] outputImage = \
            np.empty((self.iterator._numberOfWindowsVertically,
                     self.iterator._numberOfWindowsHorizontally,
                     hog.descriptorLengthPerWindow), order='F')
        cdef np.ndarray[int, ndim=3, mode='fortran'] windowsCenters \
            = \
            np.empty((self.iterator._numberOfWindowsVertically,
                     self.iterator._numberOfWindowsHorizontally, 2),
                     order='F', dtype=np.int32)
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], hog)
        del hog
        return outputImage, windowsCenters
