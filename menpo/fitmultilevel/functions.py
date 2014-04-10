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
    source: :class:`pybug.shape.PointCloud`
        The source pointcloud instance used in the alignment

    target: :class:`pybug.shape.PointCloud`
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
    noisy_transform : :class: `pybug.transform.Similarity`
        The noisy Similarity Transform
    """
    transform = AlignmentSimilarity(source, target, rotation=rotation)
    parameters = transform.as_vector()
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    parameters += noise
    return Similarity.identity(source.n_dims).from_vector(parameters)


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
