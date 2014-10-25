from __future__ import division
import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Similarity, AlignmentSimilarity


# TODO: document me
def build_sampling_grid(patch_shape):
    r"""
    """
    patch_shape = np.array(patch_shape)
    patch_half_shape = np.require(np.round(patch_shape / 2), dtype=int)
    start = -patch_half_shape
    end = patch_half_shape + 1
    sampling_grid = np.mgrid[start[0]:end[0], start[1]:end[1]]
    return sampling_grid.swapaxes(0, 2).swapaxes(0, 1)


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
    return Similarity.identity(source.n_dims).from_vector(parameters + noise)


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
    gt_points = ground_truth.points
    target_points = target.points

    if error_type == 'me_norm':
        return _compute_me_norm(target_points, gt_points)
    elif error_type == 'me':
        return _compute_me(target_points, gt_points)
    elif error_type == 'rmse':
        return _compute_rmse(target_points, gt_points)
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


def compute_cumulative_error(errors, x_axis):
    r"""
    """
    n_errors = len(errors)
    cumulative_error = [np.count_nonzero([errors <= x])
                        for x in x_axis]
    return np.array(cumulative_error) / n_errors
