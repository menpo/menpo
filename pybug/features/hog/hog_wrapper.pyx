# distutils: language = c++
# distutils: sources = pybug/features/hog/cpp/HOG.cpp

import numpy as np
cimport numpy as np
import cython
cimport cython
from libcpp.string cimport string


cdef extern from "math.h":
    double ceil(double)
    double round(double)

cdef extern from "cpp/HOG.h":
    cdef struct wins:
        int numberOfWindowsHorizontally
        int numberOfWindowsVertically
        int numberOfWindows
        int windowHeight
        int windowWidth
        int windowStepHorizontal
        int windowStepVertical
        int windowSize[2]
        int numberOfBlocksPerWindowHorizontally
        int numberOfBlocksPerWindowVertically
        int descriptorLengthPerBlock
        int descriptorLengthPerWindow
        int imageSize[2]
        int imageHeight
        int imageWidth
        unsigned int returnOnlyWindowsWithinImageLimits
        unsigned int inputImageIsGrayscale

    cdef wins WindowsInformation(double *options, int imageHeight,
                                 int imageWidth,
                                 unsigned int inputImageIsGrayscale)

    cdef void PrintInformation(double *options, wins info)

    cdef void MainLoop(double *options, wins info, double *windowImage,
                       double *descriptorMatrix, double *descriptorVector,
                       double *inputImage,
                       double *WindowsMatrixDescriptorsMatrix,
                       double *WindowsCentersMatrix)


cpdef _hog(np.ndarray[np.float64_t, ndim=3, mode='fortran'] image,
           double[:] options):
    cdef int hist1
    cdef int hist2
    # Only 1 channel means a grayscale image
    cdef unsigned int is_grayscale = image.shape[2] == 1
    cdef int imageHeight = image.shape[0]
    cdef int imageWidth = image.shape[1]
    cdef double[:] descriptorVector
    cdef double[:, :, :] descriptorMatrix
    cdef double[:, :, :] windowImage

    cdef wins info = WindowsInformation(&options[0], imageHeight,
                                        imageWidth, is_grayscale)

    # Initialize Window Image
    if is_grayscale == 1:
        windowImage = np.zeros([info.windowHeight, info.windowWidth, 1],
                               order='F')
    elif is_grayscale == 1 and options[8] == 2:
        image = np.tile(image, [1, 1, 3])
    else:
        windowImage = np.zeros([info.windowHeight, info.windowWidth, 3],
                               order='F')
    
    # Initialize descriptor vector/matrix
    cdef double binsSize = ( 1 + (options[12] == 1)) * np.pi / options[9]

    if options[8] == 1:  # dalaltriggs
        info.descriptorLengthPerBlock = <int>(options[11] * options[11] *
                                              options[9])
        hist1 = <int>(2 + ceil(-0.5 + info.windowHeight / options[10]))
        hist2 = <int>(2 + ceil(-0.5 + info.windowWidth / options[10]))

        info.numberOfBlocksPerWindowVertically = <int>(hist1 - 2 - (options[11] - 1))
        info.numberOfBlocksPerWindowHorizontally = <int>(hist2 - 2 - (options[11] - 1))
        info.descriptorLengthPerWindow = <int>(info.numberOfBlocksPerWindowVertically *
                                               info.numberOfBlocksPerWindowHorizontally *
                                               info.descriptorLengthPerBlock)

        descriptorVector = np.zeros([info.descriptorLengthPerWindow], order='F')
        descriptorMatrix = np.zeros([info.numberOfBlocksPerWindowVertically,
                                     info.numberOfBlocksPerWindowHorizontally,
                                     info.descriptorLengthPerBlock], order='F')
    elif options[8] == 2:  # zhuramanan
        hist1 = <int>round(<double>info.windowHeight / <double>options[10])
        hist2 = <int>round(<double>info.windowWidth / <double>options[10])
        descriptorMatrix = np.zeros([np.max(hist1 - 2, 0), np.max(hist2 - 2, 0), 31],
                                    order='F')
        # Unused
        descriptorVector = np.empty(1)
        info.numberOfBlocksPerWindowHorizontally = descriptorMatrix.shape[1]
        info.numberOfBlocksPerWindowVertically = descriptorMatrix.shape[0]
        info.descriptorLengthPerBlock = descriptorMatrix.shape[2]
        info.descriptorLengthPerWindow = (info.numberOfBlocksPerWindowHorizontally *
                                          info.numberOfBlocksPerWindowVertically *
                                          info.descriptorLengthPerBlock)

    # Initialize output matrices
    cdef double[:, :, :, :, :] windows_matrix_descriptors_matrix = np.zeros([info.numberOfWindowsVertically,
                                                                             info.numberOfWindowsHorizontally,
                                                                             info.numberOfBlocksPerWindowVertically,
                                                                             info.numberOfBlocksPerWindowHorizontally,
                                                                             info.descriptorLengthPerBlock], order='F')

    cdef double[:, :, :] windows_centers_matrix = np.zeros([info.numberOfWindowsVertically,
                                                            info.numberOfWindowsHorizontally,
                                                            2], order='F')

    # Processing
    MainLoop(&options[0], info, &windowImage[0, 0, 0],
             &descriptorMatrix[0, 0, 0], &descriptorVector[0],
             &image[0, 0, 0], &windows_matrix_descriptors_matrix[0, 0, 0, 0, 0],
             &windows_centers_matrix[0, 0, 0])

    return (windows_matrix_descriptors_matrix,
            windows_centers_matrix,
            _wins_to_option_information(info, options))


cdef _wins_to_option_information(wins w, double[:] options):
    return HOGOptionInformation(
        options[0],
        options[8],
        w.numberOfWindowsHorizontally,
        w.numberOfWindowsVertically,
        w.numberOfWindows,
        w.windowHeight,
        w.windowWidth,
        w.windowStepHorizontal,
        w.windowStepVertical,
        w.numberOfBlocksPerWindowHorizontally,
        w.numberOfBlocksPerWindowVertically,
        w.descriptorLengthPerBlock,
        w.descriptorLengthPerWindow,
        w.imageHeight,
        w.imageWidth,
        w.returnOnlyWindowsWithinImageLimits,
        w.inputImageIsGrayscale
    )


class HOGOptionInformation:

    def __init__(self, type, method, horizontal_window_count,
                 vertical_window_count, n_windows,window_height,
                 window_width, horizontal_window_step, vertical_window_step,
                 horizontal_window_block_count,
                 vertical_window_block_count, hog_length_per_block,
                 hog_length_per_window, image_height, image_width,
                 is_padding_disabled, is_greyscale):
        self.type = 'sparse' if type == 1.0 else 'dense'
        self.method = 'dalaltriggs' if method == 1.0 else 'zhuramanan'
        self.horizontal_window_count = horizontal_window_count
        self.vertical_window_count = vertical_window_count
        self.n_windows = n_windows
        self.window_height = window_height
        self.window_width = window_width
        self.horizontal_window_step = horizontal_window_step
        self.vertical_window_step = vertical_window_step
        self.horizontal_window_block_count = horizontal_window_block_count
        self.vertical_window_block_count = vertical_window_block_count
        self.hog_length_per_block = hog_length_per_block
        self.hog_length_per_window = hog_length_per_window
        self.image_height = image_height
        self.image_width = image_width
        self.padding_disabled = bool(is_padding_disabled)
        self.color = "greyscale" if is_greyscale else "rgb"

    def __str__(self):
        return """Input image: {0} x {1} pixels ({2})
{3} HOGs
Windows Size = {4} x {5} pixels
Windows Step = {6} x {7} pixels
Number of Windows = {8} x {9} = {10}
Descriptor per window = {11} x {12} x {13} = {14} x 1
Method of {15}
Only Windows Within Image Limits: {16}""".format(
            self.image_height, self.image_width, self.color,
            self.type,
            self.window_height, self.window_width,
            self.vertical_window_step, self.horizontal_window_step,
            self.vertical_window_count, self.horizontal_window_count,
            self.n_windows,
            self.vertical_window_block_count,
            self.horizontal_window_block_count,
            self.hog_length_per_block, self.hog_length_per_window,
            self.method,
            self.padding_disabled)
