# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_augmented_centres(const double[:, :] centres,
                                 const np.int64_t[:, :] sample_offsets,
                                 np.int64_t[:, :] augmented_centres):
    r"""
    For each centre that was given (centre of a patch), generate another
    patch that is an offset away from that centre. This is useful for generating
    a dense set of patch based features around a given point.

    Parameters (Inputs)
    -------------------
    centres : double[:, :] (n_points, 2)
        The centres of each patch.
    sample_offsets : np.int64_t[:, :] (n_sample_offsets, 2)
        The 2D offsets to sample extra patches around

    Parameters (Outputs)
    --------------------
    augmented_centres : np.int64_t[:, :] (n_points * n_sample_offsets, 2)
        The output buffer
    """
    cdef:
        np.int64_t total_index = 0, i = 0, j = 0

    for i in range(centres.shape[0]):
        for j in range(sample_offsets.shape[0]):
            augmented_centres[total_index, 0] = <np.int64_t> (centres[i, 0] + sample_offsets[j, 0])
            augmented_centres[total_index, 1] = <np.int64_t> (centres[i, 1] + sample_offsets[j, 1])
            total_index += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void calc_slices(const np.int64_t[:, :] centres,
                      np.int64_t image_shape0,
                      np.int64_t image_shape1,
                      np.int64_t patch_shape0,
                      np.int64_t patch_shape1,
                      np.int64_t[:, :] ext_s_min,
                      np.int64_t[:, :] ext_s_max,
                      np.int64_t[:, :] ins_s_min,
                      np.int64_t[:, :] ins_s_max):
    r"""
    For each centre that was given (centre of a patch), generate a slice in to
    the original image that represents the bounds of the patch. This method
    also handles ensuring that all patches have a correct set of bounds. Patches
    that would lie outside that bounds of the image are truncated back inside
    the image.

    Parameters (Inputs)
    -------------------
    centres : double[:, :] (n_points, 2)
        The centres of each patch.
    image_shape0 : np.int64_t
        The size of the first dimension of the image (height)
    image_shape1 : np.int64_t
        The size of the second dimension of the image (width)
    patch_shape0 : np.int64_t
        The size of the first dimension of the patch (height)
    patch_shape1 : np.int64_t
        The size of the second dimension of the patch (width)

    Parameters (Outputs)
    --------------------
    ext_s_min : np.int64_t[:, :] (n_centres, 2)
        The extraction slice minimum indices. This is in the image domain, one
        for each slice.
    ext_s_max : np.int64_t[:, :] (n_centres, 2)
        The extraction slice maximum indices. This is in the image domain, one
        for each slice.
    ins_s_min : np.int64_t[:, :] (n_centres, 2)
        The insertion slice minimum indices. This is in the patch domain, one
        for each slice.
    ins_s_max : np.int64_t[:, :] (n_centres, 2)
        The insertion slice maximum indices. This is in the patch domain, one
        for each slice.
    """
    cdef:
        np.int64_t c_min_new0 = 0, c_min_new1 = 0, c_max_new0 = 0, c_max_new1 = 0, i = 0
        np.int64_t half_patch_shape0 = patch_shape0 / 2
        np.int64_t half_patch_shape1 = patch_shape1 / 2
        np.int64_t add_to_patch0 = patch_shape0 % 2
        np.int64_t add_to_patch1 = patch_shape1 % 2

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
cdef void slice_image(const double[:, :, :] image,
                      const np.int64_t n_centres,
                      const np.int64_t n_sample_offsets,
                      const np.int64_t[:, :] ext_s_min,
                      const np.int64_t[:, :] ext_s_max,
                      const np.int64_t[:, :] ins_s_min,
                      const np.int64_t[:, :] ins_s_max,
                      double[:, :, :, :] patches):
    r"""
    Extract all the patches from the image. The patch extents have already been
    calculated and so this function simply slices appropriately in to the image
    to extract the pixels.

    Parameters (Inputs)
    -------------------
    image : double[:, :, :] (height, width, n_channels)
        The image to extract patches from.
    n_centres : np.int64_t
        The number of centres given.
    n_sample_offsets : np.int64_t
        The number of sample offsets given.
    ext_s_min : np.int64_t[:, :] (n_centres, 2)
        The extraction slice minimum indices. This is in the image domain, one
        for each slice.
    ext_s_max : np.int64_t[:, :] (n_centres, 2)
        The extraction slice maximum indices. This is in the image domain, one
        for each slice.
    ins_s_min : np.int64_t[:, :] (n_centres, 2)
        The insertion slice minimum indices. This is in the patch domain, one
        for each slice.
    ins_s_max : np.int64_t[:, :] (n_centres, 2)
        The insertion slice maximum indices. This is in the patch domain, one
        for each slice.

    Parameters (Outputs)
    --------------------
    patches : np.int64_t[:, :, :, :] (n_centres * n_sample_offsets, height, width, channels)
        The set of patches that have been extracted.
    """
    cdef np.int64_t i = 0
    for i in range(n_centres * n_sample_offsets):
            patches[i,
                    ins_s_min[i, 0]:ins_s_max[i, 0],
                    ins_s_min[i, 1]:ins_s_max[i, 1]] = \
            image[ext_s_min[i, 0]:ext_s_max[i, 0],
                  ext_s_min[i, 1]:ext_s_max[i, 1]]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef extract_patches_cython(double[:, :, :] image, double[:, :] centres,
                             np.int64_t[:] patch_size, np.int64_t[:, :] sample_offsets):
    r"""
    Extract a set of patches from an image. Given a set of patch centres and
    a patch size, patches are extracted from within the image centreed
    on the given coordinates. Sample offsets denote a set of offsets to extract
    from within a patch. This is very useful if you want to extract a dense
    set of features around a set of landmarks and simply sample the same grid
    of patches around the landmarks.

    Parameters
    ----------
    image : double[:, :, :] (height, width, n_channels)
        The image to extract patches from.
    n_centres : np.int64_t
        The number of centres given.
    n_sample_offsets : np.int64_t
        The number of sample offsets given.
    ext_s_min : np.int64_t[:, :] (n_centres, 2)
        The extraction slice minimum indices. This is in the image domain, one
        for each slice.
    ext_s_max : np.int64_t[:, :] (n_centres, 2)
        The extraction slice maximum indices. This is in the image domain, one
        for each slice.
    ins_s_min : np.int64_t[:, :] (n_centres, 2)
        The insertion slice minimum indices. This is in the patch domain, one
        for each slice.
    ins_s_max : np.int64_t[:, :] (n_centres, 2)
        The insertion slice maximum indices. This is in the patch domain, one
        for each slice.

    Returns
    -------
    patches : double[:, :, :, :] (n_centres * n_sample_offsets, height, width, n_channels)
        Returns a matrix containing the set of patches. The last 3 dimensions
        are the patches and the first dimension is each centre. If multiple
        sample offsets were provided, then they are all concatenated together
        to return (n_centres * n_sample_offsets) patches.
    """
    cdef:
        np.int64_t n_centres = centres.shape[0]
        np.int64_t n_sample_offsets = sample_offsets.shape[0]
        np.int64_t n_augmented_centres = n_centres * n_sample_offsets

        np.int64_t[:, :] augmented_centres = np.empty([n_augmented_centres, 2],
                                                dtype=np.int64)

        np.int64_t patch_shape0 = patch_size[0]
        np.int64_t patch_shape1 = patch_size[1]
        np.int64_t image_shape0 = image.shape[0]
        np.int64_t image_shape1 = image.shape[1]
        np.int64_t n_channels = image.shape[2]

        # This could be faster with malloc
        np.int64_t[:,:] ext_s_max = np.empty([n_augmented_centres, 2], dtype=np.int64)
        np.int64_t[:,:] ext_s_min = np.empty([n_augmented_centres, 2], dtype=np.int64)
        np.int64_t[:,:] ins_s_max = np.empty([n_augmented_centres, 2], dtype=np.int64)
        np.int64_t[:,:] ins_s_min = np.empty([n_augmented_centres, 2], dtype=np.int64)

        # It is important this array is zeros and not empty due to truncating
        # out of bounds patches.
        np.ndarray[double, ndim=4] patches = np.zeros(
            [n_centres * n_sample_offsets, patch_shape0, patch_shape1,
             n_channels])

    calc_augmented_centres(centres, sample_offsets, augmented_centres)
    calc_slices(augmented_centres,
                image_shape0,
                image_shape1,
                patch_shape0,
                patch_shape1,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max)

    slice_image(image,
                n_centres,
                n_sample_offsets,
                ext_s_min,
                ext_s_max,
                ins_s_min,
                ins_s_max,
                patches)

    return patches
