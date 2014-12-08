# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_augmented_centers(double[:,:] centres, long[:,:] offsets, long[:,:] augmented_centers):
    cdef long total_index = 0
    for i in range(centres.shape[0]):
        for j in range(offsets.shape[0]):
            augmented_centers[total_index, 0] = <long> (centres[i, 0] + offsets[j, 0])
            augmented_centers[total_index, 1] = <long> (centres[i, 1] + offsets[j, 1])
            total_index += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_slices(long[:, :] centres,
                      long image_shape0,
                      long image_shape1,
                      long patch_shape0,
                      long patch_shape1,
                      long half_patch_shape0,
                      long half_patch_shape1,
                      long add_to_patch0,
                      long add_to_patch1,
                      long[:, :] ext_s_min,
                      long[:, :] ext_s_max,
                      long[:, :] ins_s_min,
                      long[:, :] ins_s_max):
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
cdef void slice_image(double[:, :, :] image,
                      long n_channels,
                      long n_centres,
                      long n_offsets,
                      long[:, :] ext_s_min,
                      long[:, :] ext_s_max,
                      long[:, :] ins_s_min,
                      long[:, :] ins_s_max,
                      double[:, :, :, :, :] patches):
    cdef long total_index = 0

    for i in range(n_centres):
        for j in range(n_offsets):
            patches[i,
                    j,
                    :,
                    ins_s_min[total_index, 0]:ins_s_max[total_index, 0],
                    ins_s_min[total_index, 1]:ins_s_max[total_index, 1]
            ] = \
            image[:,
                  ext_s_min[total_index, 0]:ext_s_max[total_index, 0],
                  ext_s_min[total_index, 1]:ext_s_max[total_index, 1]]
            total_index += 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extract_patches(double[:,:,:] image, double[:,:] centres, long[:] patch_shape, long[:,:] offsets):
    cdef:
        long n_centres = centres.shape[0]
        long n_offsets = offsets.shape[0]
        long n_augmented_centres = n_centres * n_offsets

        long[:, :] augmented_centers = np.empty([n_augmented_centres, 2], dtype=int)

        long half_patch_shape0 = patch_shape[0] / 2
        long half_patch_shape1 = patch_shape[1] / 2
        long add_to_patch0 = patch_shape[0] % 2
        long add_to_patch1 = patch_shape[1] % 2
        long patch_shape0 = patch_shape[0]
        long patch_shape1 = patch_shape[1]
        long image_shape0 = image.shape[1]
        long image_shape1 = image.shape[2]
        long n_channels = image.shape[0]

        # This could be faster with malloc
        long[:,:] ext_s_max = np.empty([n_augmented_centres, 2], dtype=int)
        long[:,:] ext_s_min = np.empty([n_augmented_centres, 2], dtype=int)
        long[:,:] ins_s_max = np.empty([n_augmented_centres, 2], dtype=int)
        long[:,:] ins_s_min = np.empty([n_augmented_centres, 2], dtype=int)

        np.ndarray[double, ndim=5] patches = np.zeros([n_centres,
                                                       n_offsets,
                                                       n_channels,
                                                       patch_shape0,
                                                       patch_shape1])

    calc_augmented_centers(centres, offsets, augmented_centers)
    calc_slices(augmented_centers,
                image_shape0,
                image_shape1,
                patch_shape0,
                patch_shape1,
                half_patch_shape0,
                half_patch_shape1,
                add_to_patch0,
                add_to_patch1,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max)
    slice_image(image,
                n_channels,
                n_centres,
                n_offsets,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max,
                patches)

    return patches