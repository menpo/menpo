# distutils: language = c++
# distutils: sources = menpo/features/cpp/ImageWindowIterator.cpp menpo/features/cpp/WindowFeature.cpp menpo/features/cpp/HOG.cpp

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
                            unsigned int imageWidth,
                            unsigned int numberOfChannels,
                            unsigned int windowHeight,
                            unsigned int windowWidth,
                            unsigned int windowStepHorizontal,
                            unsigned int windowStepVertical,
                            bool enablePadding)
        void apply(double *outputImage, int *windowsCenters,
                   WindowFeature *windowFeature)
        unsigned int _numberOfWindowsHorizontally, \
            _numberOfWindowsVertically, _numberOfWindows, _imageWidth, \
            _imageHeight, _numberOfChannels, _windowHeight, _windowWidth, \
            _windowStepHorizontal, _windowStepVertical
        bool _enablePadding

cdef extern from "cpp/WindowFeature.h":
    cdef cppclass WindowFeature:
        void apply(double *windowImage, double *descriptorVector)
        unsigned int descriptorLengthPerWindow


cdef extern from "cpp/HOG.h":
    cdef cppclass HOG(WindowFeature):
        HOG(unsigned int windowHeight, unsigned int windowWidth,
            unsigned int numberOfChannels, unsigned int method,
            unsigned int numberOfOrientationBins,
            unsigned int cellHeightAndWidthInPixels,
            unsigned int blockHeightAndWidthInCells,
            bool enableSignedGradients, double l2normClipping)
        void apply(double *windowImage, double *descriptorVector)
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
        self.iterator = new ImageWindowIterator(&image_f[0, 0, 0],
                                                image.shape[0], image.shape[1],
                                                image.shape[2], windowHeight,
                                                windowWidth,
                                                windowStepHorizontal,
                                                windowStepVertical,
                                                enablePadding)
        if self.iterator._numberOfWindowsHorizontally == 0 or \
                        self.iterator._numberOfWindowsVertically == 0:
            raise ValueError("The window-related options are wrong. "
                             "The number of windows is 0.")

    def __str__(self):
        info_str = "Window Iterator:\n" \
                   "  - Input image is {}W x {}H with {} channels.\n" \
                   "  - Window of size {}W x {}H and step ({}W,{}H).\n"\
            .format(<int>self.iterator._imageWidth,
                    <int>self.iterator._imageHeight,
                    <int>self.iterator._numberOfChannels,
                    <int>self.iterator._windowWidth,
                    <int>self.iterator._windowHeight,
                    <int>self.iterator._windowStepHorizontal,
                    <int>self.iterator._windowStepVertical)
        if self.iterator._enablePadding:
            info_str = "{}  - Padding is enabled.\n".format(info_str)
        else:
            info_str = "{}  - Padding is disabled.\n".format(info_str)
        info_str = "{}  - Number of windows is {}W x {}H."\
            .format(info_str, <int>self.iterator._numberOfWindowsHorizontally,
                    <int>self.iterator._numberOfWindowsVertically)
        return info_str

    def HOG(self, method, numberOfOrientationBins, cellHeightAndWidthInPixels,
            blockHeightAndWidthInCells, enableSignedGradients,
            l2normClipping, verbose):

        cdef HOG *hog = new HOG(self.iterator._windowHeight,
                                self.iterator._windowWidth,
                                self.iterator._numberOfChannels, method,
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
        if verbose:
            info_str = "HOG features:\n"
            if method == 1:
                info_str = "{0}  - Algorithm of Dalal & Triggs.\n" \
                           "  - Cell is {1}x{1} pixels.\n" \
                           "  - Block is {2}x{2} cells.\n"\
                    .format(info_str, <int>cellHeightAndWidthInPixels,
                            <int>blockHeightAndWidthInCells)
                if enableSignedGradients:
                    info_str = "{}  - {} orientation bins and signed " \
                               "angles.\n"\
                        .format(info_str, <int>numberOfOrientationBins)
                else:
                    info_str = "{}  - {} orientation bins and unsigned " \
                               "angles.\n"\
                        .format(info_str, <int>numberOfOrientationBins)
                info_str = "{0}  - L2-norm clipped at {1:.1}.\n" \
                           "  - Number of blocks per window = {2}W x {3}H.\n" \
                           "  - Descriptor length per window = " \
                           "{2}W x {3}H x {4} = {5} x 1.\n"\
                    .format(info_str, l2normClipping,
                            <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically,
                            <int>hog.descriptorLengthPerBlock,
                            <int>hog.descriptorLengthPerWindow)
            else:
                info_str = "{0}  - Algorithm of Zhu & Ramanan.\n" \
                           "  - Cell is {1}x{1} pixels.\n" \
                           "  - Block is {2}x{2} cells.\n"\
                           "  - Number of blocks per window = {3}W x {4}H.\n" \
                           "  - Descriptor length per window = " \
                           "{3}W x {4}H x {5} = {6} x 1.\n"\
                    .format(info_str, <int>cellHeightAndWidthInPixels,
                            <int>blockHeightAndWidthInCells,
                            <int>hog.numberOfBlocksPerWindowHorizontally,
                            <int>hog.numberOfBlocksPerWindowVertically,
                            <int>hog.descriptorLengthPerBlock,
                            <int>hog.descriptorLengthPerWindow)
            info_str = "{}Output image size {}W x {}H x {}."\
                .format(info_str,
                        <int>self.iterator._numberOfWindowsHorizontally,
                        <int>self.iterator._numberOfWindowsVertically,
                        <int>hog.descriptorLengthPerWindow)
            print info_str
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], hog)
        del hog
        return outputImage, windowsCenters
