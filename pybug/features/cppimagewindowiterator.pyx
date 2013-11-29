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
        ImageWindowIterator(double *image, unsigned int imageHeight,
                            unsigned int imageWidth, unsigned int windowHeight,
                            unsigned int windowWidth,
                            unsigned int windowStepHorizontal,
                            unsigned int windowStepVertical,
                            bool enablePadding, bool imageIsGrayscale)
        void apply(double *outputImage, int *windowsCenters,
                   WindowFeature *windowFeature)
        unsigned int _numberOfWindowsHorizontally, _numberOfWindowsVertically, _numberOfWindows, _imageWidth, \
            _imageHeight, _windowHeight, _windowWidth, _windowStepHorizontal, _windowStepVertical, _numberOfChannels
        bool _enablePadding, _imageIsGrayscale

cdef extern from "cpp/WindowFeature.h":
    cdef cppclass WindowFeature:
        void apply(double *windowImage, bool imageIsGrayscale,
                   double *descriptorVector)
        unsigned int descriptorLengthPerWindow


cdef extern from "cpp/HOG.h":
    cdef cppclass HOG(WindowFeature):
        HOG(unsigned int windowHeight, unsigned int windowWidth,
            unsigned int method, unsigned int numberOfOrientationBins,
            unsigned int cellHeightAndWidthInPixels,
            unsigned int blockHeightAndWidthInCells,
            bool enableSignedGradients, double l2normClipping)
        void apply(double *windowImage, bool imageIsGrayscale,
                   double *descriptorVector)
        unsigned int descriptorLengthPerBlock, \
            numberOfBlocksPerWindowHorizontally, \
            numberOfBlocksPerWindowVertically

cdef class CppImageWindowIterator:
    cdef ImageWindowIterator* iterator

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] image,
                  unsigned int windowHeight, unsigned int windowWidth,
                  unsigned int windowStepHorizontal,
                  unsigned int windowStepVertical, bool enablePadding):
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] image_f = \
            np.require(image, requirements='F')
        imageIsGrayscale = image.shape[2] == 1
        self.iterator = new ImageWindowIterator(&image_f[0, 0, 0],
                                                image.shape[0], image.shape[1],
                                                windowHeight, windowWidth,
                                                windowStepHorizontal,
                                                windowStepVertical,
                                                enablePadding,
                                                imageIsGrayscale)
        if self.iterator._numberOfWindowsHorizontally == 0 or self.iterator._numberOfWindowsVertically == 0:
            raise ValueError("The window-related options are wrong. The number of windows is 0.")

    def __str__(self):
        info_str = 'Window Iterator:\n  - Input image is %dW x %dH with %d channels.\n' % \
                   (<int>self.iterator._imageWidth,
                    <int>self.iterator._imageHeight,
                    <int>self.iterator._numberOfChannels)
        info_str = '%s  - Window of size %uW x %uH and step (%dW,%dH).\n' % \
                   (info_str, <int>self.iterator._windowWidth,
                    <int>self.iterator._windowHeight,
                    <int>self.iterator._windowStepHorizontal,
                    <int>self.iterator._windowStepVertical)
        if self.iterator._enablePadding:
            info_str = '%s  - Padding is enabled.\n' % info_str
        else:
            info_str = '%s  - Padding is disabled.\n' % info_str
        info_str = '%s  - Number of windows is %dW x %dH.' % \
                   (info_str, <int>self.iterator._numberOfWindowsHorizontally,
                    <int>self.iterator._numberOfWindowsVertically)
        return info_str

    def HOG(self, method, numberOfOrientationBins, cellHeightAndWidthInPixels,
            blockHeightAndWidthInCells, enableSignedGradients,
            l2normClipping, verbose):

        cdef HOG *hog = new HOG(self.iterator._windowHeight, self.iterator._windowWidth, method,
                                numberOfOrientationBins,
                                cellHeightAndWidthInPixels,
                                blockHeightAndWidthInCells,
                                enableSignedGradients, l2normClipping)
        if hog.numberOfBlocksPerWindowVertically == 0 or hog.numberOfBlocksPerWindowHorizontally == 0:
            raise ValueError("The window-related options are wrong. The number of blocks per window is 0.")
        cdef double[:, :, :] outputImage = np.zeros([self.iterator._numberOfWindowsVertically,
                                                     self.iterator._numberOfWindowsHorizontally,
                                                     hog.descriptorLengthPerWindow], order='F')
        cdef int[:, :, :] windowsCenters = np.zeros([self.iterator._numberOfWindowsVertically,
                                                     self.iterator._numberOfWindowsHorizontally,
                                                     2], order='F', dtype=np.int32)
        #cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] outputImage = \
        #    np.empty((self.iterator._numberOfWindowsVertically,
        #             self.iterator._numberOfWindowsHorizontally,
        #             hog.descriptorLengthPerWindow), order='F')
        #cdef np.ndarray[int, ndim=3, mode='fortran'] windowsCenters \
        #    = \
        #    np.empty((self.iterator._numberOfWindowsVertically,
        #             self.iterator._numberOfWindowsHorizontally, 2),
        #             order='F', dtype=np.int32)
        info_str = 'HOG features:\n'
        if verbose:
            if method == 1:
                info_str = '%s  - Algorithm of Dalal & Triggs.\n' % info_str
                info_str = '%s  - Cell is %dx%d pixels.\n' % \
                           (info_str, <int>cellHeightAndWidthInPixels,
                            <int>cellHeightAndWidthInPixels)
                info_str = '%s  - Block is %dx%d cells.\n' % \
                           (info_str, <int>blockHeightAndWidthInCells,
                            <int>blockHeightAndWidthInCells)
                if enableSignedGradients:
                    info_str = '%s  - %d orientation bins and signed ' \
                               'angles.\n' % (info_str,
                                              <int>numberOfOrientationBins)
                else:
                    info_str = '%s  - %d orientation bins and unsigned ' \
                               'angles.\n' % (info_str,
                                              <int>numberOfOrientationBins)
                info_str = '%s  - L2-norm clipped at %.1f\n' % (info_str,
                                                                l2normClipping)
                info_str = '%s  - Number of blocks per window = %dW x %dH.\n' % \
                           (info_str, <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically)
                info_str = '%s  - Descriptor length per window = %dW x %dH x ' \
                           '%d = %d x 1.\n' % \
                           (info_str,
                            <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically,
                            <int>hog.descriptorLengthPerBlock,
                            <int>hog.descriptorLengthPerWindow)
            else:
                info_str = '%s  - Algorithm of Zhu & Ramanan.\n' % info_str
                info_str = '%s  - Cell is %dx%d pixels.\n' % \
                           (info_str, <int>cellHeightAndWidthInPixels,
                            <int>cellHeightAndWidthInPixels)
                info_str = '%s  - Block is %dx%d cells.\n' % \
                           (info_str, <int>blockHeightAndWidthInCells,
                            <int>blockHeightAndWidthInCells)
                info_str = '%s  - Number of blocks per window = %dW x %dH.\n' % \
                           (info_str, <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically)
                info_str = '%s  - Descriptor length per window = %dW x %dH x ' \
                           '%d = %d x 1.\n' % \
                           (info_str,
                            <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically,
                            <int>hog.descriptorLengthPerBlock,
                            <int>hog.descriptorLengthPerWindow)
            info_str = '%sOutput image size %dW x %dH x %d.' \
                       % (info_str,
                          <int>self.iterator._numberOfWindowsHorizontally,
                          <int>self.iterator._numberOfWindowsVertically,
                          <int>hog.descriptorLengthPerWindow)
            print info_str
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], hog)
        del hog
        return outputImage, windowsCenters