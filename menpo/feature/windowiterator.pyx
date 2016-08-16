import numpy as np
cimport numpy as np
from libcpp cimport bool
from collections import namedtuple

WindowIteratorResult = namedtuple('WindowInteratorResult', ('pixels',
                                                            'centres'))

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

cdef extern from "cpp/LBP.h":
    cdef cppclass LBP(WindowFeature):
        LBP(unsigned int windowHeight, unsigned int windowWidth,
            unsigned int numberOfChannels, unsigned int *radius,
            unsigned int *samples,
            unsigned int numberOfRadiusSamplesCombinations,
            unsigned int mapping_type, unsigned int *uniqueSamples,
            unsigned int *whichMappingTable, unsigned int numberOfUniqueSamples)
        void apply(double *windowImage, double *descriptorVector)

cdef class WindowIterator:
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
        if hog.numberOfBlocksPerWindowVertically == 0 or \
                hog.numberOfBlocksPerWindowHorizontally == 0:
            raise ValueError("The window-related options are wrong. "
                             "The number of blocks per window is 0.")
        cdef double[:, :, :] outputImage = np.zeros(
            [self.iterator._numberOfWindowsVertically,
             self.iterator._numberOfWindowsHorizontally,
             hog.descriptorLengthPerWindow], order='F')
        cdef int[:, :, :] windowsCenters = np.zeros(
            [self.iterator._numberOfWindowsVertically,
             self.iterator._numberOfWindowsHorizontally,
             2], order='F', dtype=np.int32)
        if verbose:
            info_str = "HOG features:\n"
            if method == 1:
                info_str = "{0}  - Algorithm of Dalal & Triggs.\n" \
                           "  - Cell is {1}x{1} pixels.\n" \
                           "  - Block is {2}x{2} cells.\n".format(
                    info_str, <int>cellHeightAndWidthInPixels,
                    <int>blockHeightAndWidthInCells)
                if enableSignedGradients:
                    info_str = "{}  - {} orientation bins and signed " \
                               "angles.\n".format(info_str,
                                                  <int>numberOfOrientationBins)
                else:
                    info_str = "{}  - {} orientation bins and unsigned " \
                               "angles.\n".format(info_str,
                                                  <int>numberOfOrientationBins)
                info_str = "{0}  - L2-norm clipped at {1:.1}.\n" \
                           "  - Number of blocks per window = {2}W x {3}H.\n" \
                           "  - Descriptor length per window = " \
                           "{2}W x {3}H x {4} = {5} x 1.\n".format(
                    info_str, l2normClipping,
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
                           "{3}W x {4}H x {5} = {6} x 1.\n".format(
                    info_str, <int>cellHeightAndWidthInPixels,
                    <int>blockHeightAndWidthInCells,
                    <int>hog.numberOfBlocksPerWindowHorizontally,
                    <int>hog.numberOfBlocksPerWindowVertically,
                    <int>hog.descriptorLengthPerBlock,
                    <int>hog.descriptorLengthPerWindow)
            info_str = "{}Output image size {}W x {}H x {}.".format(
                info_str, <int>self.iterator._numberOfWindowsHorizontally,
                <int>self.iterator._numberOfWindowsVertically,
                <int>hog.descriptorLengthPerWindow)
            print(info_str)
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], hog)
        del hog
        return WindowIteratorResult(np.ascontiguousarray(outputImage),
                                    np.ascontiguousarray(windowsCenters))

    def LBP(self, radius, samples, mapping_type, verbose):
        # find unique samples (thus lbp codes mappings)
        uniqueSamples, whichMappingTable = np.unique(samples,
                                                     return_inverse=True)
        numberOfUniqueSamples = uniqueSamples.size
        cdef unsigned int[:] cradius = np.ascontiguousarray(radius,
                                                            dtype=np.uint32)
        cdef unsigned int[:] csamples = np.ascontiguousarray(samples,
                                                             dtype=np.uint32)
        cdef unsigned int[:] cuniqueSamples = np.ascontiguousarray(
            uniqueSamples, dtype=np.uint32)
        cdef unsigned int[:] cwhichMappingTable = np.ascontiguousarray(
            whichMappingTable, dtype=np.uint32)
        cdef LBP *lbp = new LBP(self.iterator._windowHeight,
                                self.iterator._windowWidth,
                                self.iterator._numberOfChannels, &cradius[0],
                                &csamples[0], radius.size, mapping_type,
                                &cuniqueSamples[0], &cwhichMappingTable[0],
                                numberOfUniqueSamples)
        cdef double[:, :, :] outputImage = np.zeros(
            [self.iterator._numberOfWindowsVertically,
             self.iterator._numberOfWindowsHorizontally,
             lbp.descriptorLengthPerWindow], order='F')
        cdef int[:, :, :] windowsCenters = np.zeros(
            [self.iterator._numberOfWindowsVertically,
             self.iterator._numberOfWindowsHorizontally,
             2], order='F', dtype=np.int32)
        if verbose:
            info_str = "LBP features:\n"
            if radius.size == 1:
                info_str = "{0}  - 1 combination of radius and " \
                           "samples.\n".format(info_str)
                info_str = "{0}  - Radius value: {1}.\n".format(info_str,
                                                                <int>radius[0])
                info_str = "{0}  - Samples value: {1}.\n".format(
                    info_str, <int>samples[0])
            else:
                info_str = "{0}  - {1} combinations of radii and " \
                           "samples.\n".format(info_str, <int>radius.size)
                info_str = "{0}  - Radii values: [".format(info_str,
                                                           <int>radius.size)
                for k in range(radius.size - 1):
                    info_str = "{0}{1}, ".format(info_str, <int>radius[k])
                info_str = "{0}{1}].\n".format(info_str, <int>radius[-1])
                info_str = "{0}  - Samples values: [".format(info_str,
                                                             <int>samples.size)
                for k in range(samples.size - 1):
                    info_str = "{0}{1}, ".format(info_str, <int>samples[k])
                info_str = "{0}{1}].\n".format(info_str, <int>samples[-1])
            if mapping_type == 1:
                info_str = "{0}  - Uniform-2 codes mapping.\n".format(info_str)
            elif mapping_type == 2:
                info_str = "{0}  - Rotation-Invariant codes mapping.\n".format(
                    info_str)
            elif mapping_type == 3:
                info_str = "{0}  - Uniform-2 and Rotation-Invariant codes " \
                           "mapping.\n".format(info_str)
            elif mapping_type == 0:
                info_str = "{0}  - No codes mapping used.\n".format(info_str)
            info_str = "{0}  - Descriptor length per window = " \
                       "{1} x 1.\n".format(info_str,
                                           <int>lbp.descriptorLengthPerWindow)
            info_str = "{}Output image size {}W x {}H x {}.".format(
                info_str, <int>self.iterator._numberOfWindowsHorizontally,
                <int>self.iterator._numberOfWindowsVertically,
                <int>lbp.descriptorLengthPerWindow)
            print(info_str)
        self.iterator.apply(&outputImage[0,0,0], &windowsCenters[0,0,0], lbp)
        del lbp
        return WindowIteratorResult(np.ascontiguousarray(outputImage),
                                    np.ascontiguousarray(windowsCenters))

def _lbp_mapping_table(n_samples, mapping_type='riu2'):
    r"""
    Returns the mapping table for LBP codes in a neighbourhood of n_samples
    number of sampling points.

    Parameters
    ----------
    n_samples :  int
        The number of sampling points.
    mapping_type : 'u2' or 'ri' or 'riu2' or 'none'
        The mapping type. Select 'u2' for uniform-2 mapping, 'ri' for
        rotation-invariant mapping, 'riu2' for uniform-2 and
        rotation-invariant mapping and 'none' to use no mapping.

        Default: 'riu2'

    Raises
    -------
    ValueError
        mapping_type can be 'u2' or 'ri' or 'riu2' or 'none'.
    """
    # initialize the output lbp codes mapping table
    table = range(2**n_samples)
    # uniform-2 mapping
    if mapping_type == 'u2':
        # initialize the number of patterns in the mapping table
        new_max = n_samples * (n_samples - 1) + 3
        index = 0
        for c in range(2**n_samples):
            # number of 1->0 and 0->1 transitions in a binary string x is equal
            # to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = bin(c ^ circural_rotation_left(c, 1, n_samples)).\
                count('1')
            if num_trans <= 2:
                table[c] = index
                index += 1
            else:
                table[c] = new_max - 1
    # rotation-invariant mapping
    elif mapping_type == 'ri':
        new_max = 0
        tmp_map = np.zeros((2**n_samples, 1), dtype=np.int) - 1
        for c in range(2**n_samples):
            rm = c
            r = c
            for j in range(1, n_samples):
                r = circural_rotation_left(r, 1, n_samples)
                rm = min(rm, r)
            if tmp_map[rm, 0] < 0:
                tmp_map[rm, 0] = new_max
                new_max += 1
            table[c] = tmp_map[rm, 0]
    # uniform-2 and rotation-invariant mapping
    elif mapping_type == 'riu2':
        new_max = n_samples + 2
        for c in range(2**n_samples):
            # number of 1->0 and 0->1 transitions in a binary string x is equal
            # to the number of 1-bits in XOR(x, rotate_left(x))
            num_trans = bin(c ^ circural_rotation_left(c, 1, n_samples)).\
                count('1')
            if num_trans <= 2:
                table[c] = bin(c).count('1')
            else:
                table[c] = n_samples + 1
    elif mapping_type == 'none':
        table = 0
        new_max = 0
    else:
        raise ValueError('Wrong mapping type.')
    return table, new_max


def circural_rotation_left(val, rot_bits, max_bits):
    r"""
    Applies a circular left shift of 'rot_bits' bits on the given number 'num'
    keeping 'max_bits' number of bits.

    Parameters
    ----------
    val :  int
        The input number to be shifted.
    rot_bins : int
        The number of bits of the left circular shift.
    max_bits : int
        The number of bits of the output number. All the bits in positions
        larger than max_bits are dropped.
    """
    return (val << rot_bits % max_bits) & (2**max_bits - 1) | \
           ((val & (2**max_bits - 1)) >> (max_bits - (rot_bits % max_bits)))


def circural_rotation_right(val, rot_bits, max_bits):
    return ((val & (2**max_bits - 1)) >> rot_bits % max_bits) | \
           (val << (max_bits - (rot_bits % max_bits)) & (2**max_bits - 1))
