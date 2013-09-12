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
        HOG(unsigned int windowHeight, unsigned int windowWidth, unsigned int method, unsigned int numberOfOrientationBins, unsigned int cellHeightAndWidthInPixels,
			unsigned int blockHeightAndWidthInCells, bool enableSignedGradients, double l2normClipping)
        void apply(double *windowImage, bool imageIsGrayscale, double *descriptorVector)
        unsigned int descriptorLengthPerBlock, numberOfBlocksPerWindowHorizontally, numberOfBlocksPerWindowVertically

cdef class CppImageWindowIterator:
    cdef ImageWindowIterator* iterator

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] image,
                  unsigned int windowHeight, unsigned int windowWidth,
                  unsigned int windowStepHorizontal,
                  unsigned int windowStepVertical, bool enablePadding):
        cdef np.ndarray[np.float64_t, ndim=3, mode='fortran'] image_f = \
            np.require(image, requirements='F')
        imageIsGrayscale = image.shape[2] == 1
        self.iterator = new ImageWindowIterator(&image_f[0, 0, 0], image.shape[0], image.shape[1], windowHeight,
                                                windowWidth, windowStepHorizontal, windowStepVertical, enablePadding,
                                                imageIsGrayscale)
        if self.iterator._numberOfWindowsHorizontally == 0 or self.iterator._numberOfWindowsVertically == 0:
            raise ValueError("The window-related options are wrong. The number of windows is 0.")

    def __str__(self):
        if self.iterator._imageIsGrayscale:
            info_str = 'Input image is GRAY with size %dx%dx%d\n' % (<int>self.iterator._imageHeight, <int>self.iterator._imageWidth, <int>self.iterator._numberOfChannels)
        else:
            info_str = 'Input image is RGB with size %dx%dx%d\n' % (<int>self.iterator._imageHeight, <int>self.iterator._imageWidth, <int>self.iterator._numberOfChannels)
        info_str = '%sWindow of size %ux%u and step (%d,%d)\n' % (info_str, <int>self.iterator._windowHeight, <int>self.iterator._windowWidth,
                                                                  <int>self.iterator._windowStepVertical, <int>self.iterator._windowStepHorizontal)
        if self.iterator._enablePadding:
            info_str = '%sPadding is enabled\n' % info_str
        else:
            info_str = '%sPadding is disabled\n' % info_str
        info_str = '%sNumber of windows is %dx%d\n' % (info_str, <int>self.iterator._numberOfWindowsVertically,
                                                       <int>self.iterator._numberOfWindowsHorizontally)
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
        info_str = 'HOG features\n'
        if verbose:
            if method == 1:
                info_str = '%sMethod of Dalal & Triggs\n' % info_str
                info_str = '%sCell = %dx%d pixels\nBlock = %dx%d cells\n' % (info_str, <int>cellHeightAndWidthInPixels,
                                                                             <int>cellHeightAndWidthInPixels,
                                                                             <int>blockHeightAndWidthInCells,
                                                                             <int>blockHeightAndWidthInCells)
                if enableSignedGradients:
                    info_str = '%s%d orientation bins and signed gradients\n' % (info_str, <int>numberOfOrientationBins)
                else:
                    info_str = '%s%d orientation bins and unsigned gradients\n' % (info_str, <int>numberOfOrientationBins)
                info_str = '%sL2-norm clipped at %.1f\nNumber of blocks per window = %dx%d\n' % (info_str, l2normClipping,
                                                                                             <int>hog.numberOfBlocksPerWindowVertically,
                                                                                             <int>hog.numberOfBlocksPerWindowHorizontally)
                info_str = '%sDescriptor length per window = %dx%dx%d = %d\n' % (info_str, <int>hog.numberOfBlocksPerWindowVertically,
                                                                            <int>hog.numberOfBlocksPerWindowHorizontally, <int>hog.descriptorLengthPerBlock,
                                                                            <int>hog.descriptorLengthPerWindow)
            else:
                info_str = '%sMethod of Zhu & Ramanan\n' % info_str
                info_str = '%sCell = %dx%d pixels\nBlock = %dx%d cells\n' % (info_str, <int>cellHeightAndWidthInPixels,
                                                                             <int>cellHeightAndWidthInPixels,
                                                                             <int>blockHeightAndWidthInCells,
                                                                             <int>blockHeightAndWidthInCells)
                info_str = '%sNumber of blocks per window = %dx%d\n' % (info_str,
                                                                                             <int>hog.numberOfBlocksPerWindowVertically,
                                                                                             <int>hog.numberOfBlocksPerWindowHorizontally)
                info_str = '%sDescriptor length per window = %dx%dx%d = %d\n' % (info_str, <int>hog.numberOfBlocksPerWindowVertically,
                                                                            <int>hog.numberOfBlocksPerWindowHorizontally, <int>hog.descriptorLengthPerBlock,
                                                                            <int>hog.descriptorLengthPerWindow)
            print info_str
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], hog)
        del hog
        return outputImage, windowsCenters