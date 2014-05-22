from __future__ import division
import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Similarity, AlignmentSimilarity


# TODO: document me
def build_sampling_grid(grid_size):
    r"""
    """
    patch_size = np.array(grid_size)
    patch_half_size = (np.round(patch_size / 2)).astype(int)
    start = -patch_half_size
    end = patch_half_size + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


# TODO: document me
def extract_local_patches(image, shape, sampling_grid):
    r"""
    """
    max_x = image.shape[0] - 1
    max_y = image.shape[1] - 1

    patches = []
    for point in shape.points:
        patch_grid = (sampling_grid +
                      np.round(point[None, None, ...]).astype(int))
        x = patch_grid[:, :, 0]
        y = patch_grid[:, :, 1]

        # deal with boundaries
        x[x > max_x] = max_x
        y[y > max_y] = max_y
        x[x < 0] = 0
        y[y < 0] = 0

        patch_data = image.pixels[x, y, :]
        patch_img = image.__class__(patch_data)
        patches.append(patch_img)

    return patches


def extract_local_patches_fast(image, centres, patch_shape, out=None):
    r"""extract square patches from an image about centres.

    Parameters
    ----------

    image: :map:`Image`
        The image to extract patches from

    centres: :map:`PointCloud`
        The centres around which the patches should be extracted

    patch_shape: tuple of ints
        The size of the patch in each dimension

    out: ndarray shape patch_shape + (n_channels, ) n_centres, optional
        The output array to be assigned to. If None, a new numpy array will be
        created.

    Returns
    -------
    ndarray shape patch_shape + (n_channels, ) n_centres, optional
        The patches as a single numpy array.

    """
    if out is not None:
        patches = out
    else:
        patches = np.empty(
            (centres.n_points,) + patch_shape + (image.n_channels,))
    # 0 out the patches array
    patches[...] = 0
    image_size = np.array(image.shape, dtype=np.int)
    patch_shape = np.array(patch_shape, dtype=np.int)
    centres = np.require(centres.points, dtype=np.int)
    half_patch_shape = np.require(np.ceil(patch_shape / 2), dtype=np.int)
    # 1. compute the extents
    c_min = centres - half_patch_shape
    c_max = centres + half_patch_shape
    out_min_min = c_min < 0
    out_min_max = c_min > image_size
    out_max_min = c_max < 0
    out_max_max = c_max > image_size

    # 1. Build the extraction slices
    ext_s_min = c_min.copy()
    ext_s_max = c_max.copy()
    # Clamp the min to 0
    ext_s_min[out_min_min] = 0
    ext_s_max[out_max_min] = 0
    # Clamp the max to image bounds across each dimension
    for i in xrange(image.n_dims):
        ext_s_max[out_max_max[:, i], i] = image_size[i] - 1
        ext_s_min[out_min_max[:, i], i] = image_size[i] - 1

    # 2. Build the insertion slices
    ins_s_min = ext_s_min - c_min
    ins_s_max = np.maximum(ext_s_max - c_max + patch_shape, (0, 0))

    for i, (e_a, e_b, i_a, i_b) in enumerate(zip(ext_s_min, ext_s_max,
                                                 ins_s_min, ins_s_max)):
        # build a list of insertion slices and extraction slices
        i_slices = [slice(a, b) for a, b in zip(i_a, i_b)]
        e_slices = [slice(a, b) for a, b in zip(e_a, e_b)]
        # get a view onto the patch we are on
        patch = patches[i, ...]
        # apply the slices to map
        patch[i_slices] = image.pixels[e_slices]

    return patches


def mean_pointcloud(pointclouds):
    r"""
    Compute the mean of a list of point cloud objects

    Parameters
    ----------
    pointclouds: list of :class:`menpo.shape.PointCloud`
        List of point cloud objects from which we want to
        compute the mean.

    Returns
    -------
    mean_pointcloud: class:`menpo.shape.PointCloud`
        The mean point cloud.
    """
    return PointCloud(np.mean([pc.points for pc in pointclouds], axis=0))


# TODO: Should this be a method on Similarity? AlignableTransforms?
def noisy_align(source, target, noise_std=0.04, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    to the target by adding white noise to its weights.

    Parameters
    ----------
    source: :class:`menpo.shape.PointCloud`
        The source pointcloud instance used in the alignment
    target: :class:`menpo.shape.PointCloud`
        The target pointcloud instance used in the alignment
    noise_std: float
        The standard deviation of the white noise

        Default: 0.04
    rotation: boolean
        If False the second parameter of the Similarity,
        which captures captures inplane rotations, is set to 0.

        Default:False

    Returns
    -------
    noisy_transform : :class: `menpo.transform.Similarity`
        The noisy Similarity Transform
    """
    transform = AlignmentSimilarity(source, target, rotation=rotation)
    parameters = transform.as_vector()
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    parameters += noise
    return Similarity.identity(source.n_dims).from_vector(parameters)


def align_shape_with_bb(shape, bounding_box):
    r"""
    Returns the Similarity transform that aligns the provided shape with the
    provided bounding box.

    Parameters
    ----------
    shape: :class:`menpo.shape.PointCloud`
        The shape to be aligned.
    bounding_box: (2, 2) ndarray
        The bounding box specified as:

            np.array([[x_min, y_min], [x_max, y_max]])

    Returns
    -------
    transform : :class: `menpo.transform.Similarity`
        The align transform
    """
    shape_box = PointCloud(shape.bounds())
    bounding_box = PointCloud(bounding_box)
    return AlignmentSimilarity(shape_box, bounding_box, rotation=False)


#TODO: Document me
def compute_error(target, ground_truth, error_type='me_norm'):
    r"""
    """
    if error_type is 'me_norm':
        return _compute_me_norm(target, ground_truth)
    elif error_type is 'me':
        return _compute_me(target, ground_truth)
    elif error_type is 'rmse':
        return _compute_rmse(target, ground_truth)
    else:
        raise ValueError("Unknown error_type string selected. Valid options "
                         "are: me_norm, me, rmse'")


#TODO: Document me
def _compute_rmse(target, ground_truth):
    r"""
    """
    return np.sqrt(np.mean((target.flatten() - ground_truth.flatten()) ** 2))


#TODO: Document me
def _compute_me(target, ground_truth):
    r"""
    """
    return np.mean(np.sqrt(np.sum((target - ground_truth) ** 2, axis=-1)))


#TODO: Document me
def _compute_me_norm(target, ground_truth):
    r"""
    """
    normalizer = np.mean(np.max(ground_truth, axis=0) -
                         np.min(ground_truth, axis=0))
    return _compute_me(target, ground_truth) / normalizer
