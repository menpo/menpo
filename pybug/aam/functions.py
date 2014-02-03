from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from pybug.shape import PointCloud, TriMesh
from pybug.transform .affine import Translation, SimilarityTransform
from pybug.image import MaskedImage


def build_reference_frame(landmarks, boundary=3, group='source',
                          trilist=None):
    r"""
    Build reference frame from landmarks.

    Parameters
    ----------
    reference_landmarks:
    scale: int, optional

        Default: 1
    boundary: int, optional

        Default: 3
    group: str, optional

        Default: 'source'
    trilist: (Nt, 3) ndarray, optional

        Default: None

    Returns
    -------
    rescaled_image : type(self)
        A copy of this image, rescaled.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks['source'].lms.points, trilist=trilist)

    # TODO: revise kwarg trilist in method constrain_mask_to_landmarks,
    # perhaps the trilist should be directly obtained from the group landmarks
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_size=(16, 16)):
    r"""
    Build reference frame from landmarks.

    Parameters
    ----------
    reference_landmarks:
    scale: int, optional

        Default: 1
    boundary: int, optional

        Default: 3
    group: str, optional

        Default: 'source'
    patch_size: dictionary, optional

        Default: None

    Returns
    -------
    rescaled_image : type(self)
        A copy of this image, rescaled.
    """
    boundary = np.max(patch_size) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_size, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame


# TODO: Should this be a method on SimilarityTransform? AlignableTransforms?
def noisy_align(source, target, noise_std=0.05, rotation=False):
    r"""
    Constructs and perturbs the optimal similarity transform between source
    to the target by adding white noise to its parameters.

    Parameters
    ----------
    source: :class:`pybug.shape.PointCloud`
        The source pointcloud instance used in the alignment

    target: :class:`pybug.shape.PointCloud`
        The target pointcloud instance used in the alignment

    noise_std: float
        The standard deviation of the white noise

        Default: 0.05
    rotation: boolean
        If False the second parameter of the SimilarityTransform,
        which captures captures inplane rotations, is set to 0.

        Default:False

    Returns
    -------
    noisy_transform : :class: `pybug.transform.SimilarityTransform`
        The noisy Similarity Transform
    """
    transform = SimilarityTransform.align(source, target)
    parameters = transform.as_vector()
    if not rotation:
        parameters[1] = 0
    parameter_range = np.hstack((parameters[:2], target.range()))
    noise = (parameter_range * noise_std *
             np.random.randn(transform.n_parameters))
    parameters += noise
    return SimilarityTransform.from_vector(parameters)


def compute_features(image, features):
    r"""
    Compute feature images.

    Parameters
    ----------
    image: :class:`pybug.image.MaskedNDImage`
        The original image from which the features will be computed
    features_dic: dictionary
        ['type'] : string
            String specifying the type of features to compute
        ['options']: **kwargs:
            Passed through to the particular feature method being used. See
            `pybug.image.MaskedNDImage` for details on feature options.

    Returns
    -------
    feature_image: :class:`pybug.image.MaskedNDImage`
        The resulting feature image.
    """
    if features is not None:
        if features[0] is 'normalize_std':
            image.normalize_std_inplace(**features[1])
        elif features[0] is 'normalize_norm':
            image.normalize_norm_inplace(**features[1])
        elif features[0] is 'euler':
            raise NotImplementedError("Euler features not implemented yet")
        elif features[0] is 'igo':
            raise NotImplementedError("IGO features not implemented yet")
        elif features[0] is 'es':
            raise NotImplementedError("ES features not implemented yet")
        elif features[0] is 'hog':
            raise NotImplementedError("HoG features not implemented yet")
        elif features[0] is 'sift':
            raise NotImplementedError("Sift features not implemented yet")

    # TODO: These should disappear with the new image refactoring
    nd_image = MaskedImage(image.pixels, mask=image.mask)
    nd_image.landmarks = image.landmarks

    return nd_image


def mean_pointcloud(pointcloud_list):
    r"""
    Compute the mean of a list of Point Cloud objects

    Parameters
    ----------
    pointcloud_list: list of :class:`pybug.shape.pointcloud`
        List of PointCloud objects from which we want to
        compute the mean..

    Returns
    -------
    mean_pointcloud: class:`pybug.shape.pointcloud`
        The mean PointCloud
    """
    return PointCloud(np.mean([pc.points for pc in pointcloud_list], axis=0))


def compute_error(target, ground_truth, error_type='me_norm'):
    if error_type is 'me_norm':
        return _compute_me_norm(target, ground_truth)
    elif error_type is 'me':
        return _compute_me(target, ground_truth)
    elif error_type is 'rmse':
        return _compute_rmse(target, ground_truth)
    else:
        raise ValueError("Unknown error_type string selected. Valid options "
                         "are: me_norm, me, rmse'")


def _compute_rmse(target, ground_truth):
    return np.sqrt(np.mean((target.flatten() - ground_truth.flatten()) ** 2))


def _compute_me(target, ground_truth):
    return np.mean(np.sqrt(np.sum((target - ground_truth) ** 2, axis=-1)))


def _compute_me_norm(target, ground_truth):
    normalizer = np.mean(np.max(ground_truth, axis=0) -
                         np.min(ground_truth, axis=0))
    return _compute_me(target, ground_truth) / normalizer
