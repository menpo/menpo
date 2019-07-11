import numpy as np
cimport numpy as np
cimport cython
from ..cy_utils cimport dtype_from_memoryview


ctypedef fused IMAGE_TYPES:
    float
    double
    np.uint8_t
    np.uint16_t


ctypedef fused CENTRE_TYPES:
    float
    double


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_augmented_centers(CENTRE_TYPES[:, :] centres,
                                 Py_ssize_t[:, :] offsets,
                                 Py_ssize_t[:, :] augmented_centers):
    cdef Py_ssize_t total_index = 0, i = 0, j = 0

    for i in range(centres.shape[0]):
        for j in range(offsets.shape[0]):
            augmented_centers[total_index, 0] = <Py_ssize_t> (centres[i, 0] + offsets[j, 0])
            augmented_centers[total_index, 1] = <Py_ssize_t> (centres[i, 1] + offsets[j, 1])
            total_index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_slices(Py_ssize_t[:, :] centres,
                      Py_ssize_t image_shape0,
                      Py_ssize_t image_shape1,
                      Py_ssize_t patch_shape0,
                      Py_ssize_t patch_shape1,
                      Py_ssize_t half_patch_shape0,
                      Py_ssize_t half_patch_shape1,
                      Py_ssize_t add_to_patch0,
                      Py_ssize_t add_to_patch1,
                      Py_ssize_t[:, :] ext_s_min,
                      Py_ssize_t[:, :] ext_s_max,
                      Py_ssize_t[:, :] ins_s_min,
                      Py_ssize_t[:, :] ins_s_max):
    cdef Py_ssize_t i = 0

    for i in range(centres.shape[0]):
        c_min_new0 = centres[i, 0] - half_patch_shape0
        c_min_new1 = centres[i, 1] - half_patch_shape1
        c_max_new0 = centres[i, 0] + half_patch_shape0 + add_to_patch0
        c_max_new1 = centres[i, 1] + half_patch_shape1 + add_to_patch1

        ext_s_min[i, 0] = c_min_new0
        ext_s_min[i, 1] = c_min_new1
        ext_s_max[i, 0] = c_max_new0
        ext_s_max[i, 1] = c_max_new1

        if ext_s_min[i, 0] < 0:
            ext_s_min[i, 0] = 0
        if ext_s_min[i, 1] < 0:
            ext_s_min[i, 1] = 0
        if ext_s_min[i, 0] > image_shape0:
            ext_s_min[i, 0] = image_shape0 - 1
        if ext_s_min[i, 1] > image_shape1:
            ext_s_min[i, 1] = image_shape1 - 1

        if ext_s_max[i, 0] < 0:
            ext_s_max[i, 0] = 0
        if ext_s_max[i, 1] < 0:
            ext_s_max[i, 1] = 0
        if ext_s_max[i, 0] > image_shape0:
            ext_s_max[i, 0] = image_shape0 - 1
        if ext_s_max[i, 1] > image_shape1:
            ext_s_max[i, 1] = image_shape1 - 1

        ins_s_min[i, 0] = ext_s_min[i, 0] - c_min_new0
        ins_s_min[i, 1] = ext_s_min[i, 1] - c_min_new1

        ins_s_max[i, 0] = ext_s_max[i, 0] - c_max_new0 + patch_shape0
        if ins_s_max[i, 0] < 0:
            ins_s_max[i, 0] = 0
        ins_s_max[i, 1] = ext_s_max[i, 1] - c_max_new1 + patch_shape1
        if ins_s_max[i, 1] < 0:
            ins_s_max[i, 1] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extract_patches(IMAGE_TYPES[:, :, :] image,
                      CENTRE_TYPES[:, :] centres,
                      Py_ssize_t[:] patch_shape, Py_ssize_t[:, :] offsets):
    dtype = dtype_from_memoryview(image)
    cdef:
        Py_ssize_t n_centres = centres.shape[0]
        Py_ssize_t n_offsets = offsets.shape[0]
        Py_ssize_t n_augmented_centres = n_centres * n_offsets
        object extents_size = [n_augmented_centres, 2]

        Py_ssize_t half_patch_shape0 = patch_shape[0] // 2
        Py_ssize_t half_patch_shape1 = patch_shape[1] // 2
        Py_ssize_t add_to_patch0 = patch_shape[0] % 2
        Py_ssize_t add_to_patch1 = patch_shape[1] % 2
        Py_ssize_t patch_shape0 = patch_shape[0]
        Py_ssize_t patch_shape1 = patch_shape[1]
        Py_ssize_t image_shape0 = image.shape[1]
        Py_ssize_t image_shape1 = image.shape[2]
        Py_ssize_t n_channels = image.shape[0]

        Py_ssize_t total_index = 0, i = 0, j = 0

        # Although it is faster to use malloc in this case, the change in syntax
        # and the mental overhead of handling freeing memory is not considered
        # worth it for these buffers. From simple tests it seems you only begin
        # to see a performance difference when you have
        # n_augmented_centres >~ 5000
        Py_ssize_t[:, :] augmented_centers = np.empty(extents_size,
                                                      dtype=np.intp)
        Py_ssize_t[:, :] ext_s_max = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ext_s_min = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ins_s_max = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ins_s_min = np.empty(extents_size, dtype=np.intp)

        np.ndarray[IMAGE_TYPES, ndim=5] patches = np.zeros(
            [n_centres, n_offsets, n_channels, patch_shape0, patch_shape1],
            dtype=dtype)

    calc_augmented_centers(centres, offsets, augmented_centers)
    calc_slices(augmented_centers, image_shape0, image_shape1, patch_shape0,
                patch_shape1, half_patch_shape0, half_patch_shape1,
                add_to_patch0, add_to_patch1, ext_s_min, ext_s_max, ins_s_min,
                ins_s_max)

    for i in range(n_centres):
        for j in range(n_offsets):
            patches[i, j, :,
                ins_s_min[total_index, 0]:ins_s_max[total_index, 0],
                ins_s_min[total_index, 1]:ins_s_max[total_index, 1]
            ] = \
                image[:,
                    ext_s_min[total_index, 0]:ext_s_max[total_index, 0],
                    ext_s_min[total_index, 1]:ext_s_max[total_index, 1]]
            total_index += 1

    return patches


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void set_patches(IMAGE_TYPES[:, :, :, :, :] patches,
                       IMAGE_TYPES[:, :, :] image,
                       CENTRE_TYPES[:, :] centres,
                       Py_ssize_t[:, :] offset,
                       Py_ssize_t offset_index):
    cdef:
        Py_ssize_t n_centres = centres.shape[0]
        object extents_size = [n_centres, 2]

        Py_ssize_t patch_shape0 = patches.shape[3]
        Py_ssize_t patch_shape1 = patches.shape[4]
        Py_ssize_t half_patch_shape0 = patch_shape0 // 2
        Py_ssize_t half_patch_shape1 = patch_shape1 // 2
        Py_ssize_t add_to_patch0 = patch_shape0 % 2
        Py_ssize_t add_to_patch1 = patch_shape1 % 2
        Py_ssize_t image_shape0 = image.shape[1]
        Py_ssize_t image_shape1 = image.shape[2]
        Py_ssize_t n_channels = image.shape[0]

        Py_ssize_t total_index = 0, i = 0

        # Although it is faster to use malloc in this case, the change in syntax
        # and the mental overhead of handling freeing memory is not considered
        # worth it for these buffers. From simple tests it seems you only begin
        # to see a performance difference when you have
        # n_augmented_centres >~ 5000
        Py_ssize_t[:, :] augmented_centers = np.empty(extents_size,
                                                      dtype=np.intp)
        Py_ssize_t[:, :] ext_s_max = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ext_s_min = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ins_s_max = np.empty(extents_size, dtype=np.intp)
        Py_ssize_t[:, :] ins_s_min = np.empty(extents_size, dtype=np.intp)

    calc_augmented_centers(centres, offset, augmented_centers)
    calc_slices(augmented_centers, image_shape0, image_shape1, patch_shape0,
                patch_shape1, half_patch_shape0, half_patch_shape1,
                add_to_patch0, add_to_patch1, ext_s_min, ext_s_max, ins_s_min,
                ins_s_max)

    for i in range(n_centres):
        image[:,
              ext_s_min[total_index, 0]:ext_s_max[total_index, 0],
              ext_s_min[total_index, 1]:ext_s_max[total_index, 1]
             ] = \
                 patches[i, offset_index, :,
                         ins_s_min[total_index, 0]:ins_s_max[total_index, 0],
                         ins_s_min[total_index, 1]:ins_s_max[total_index, 1]]
        total_index += 1
