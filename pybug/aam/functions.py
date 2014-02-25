from __future__ import division
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


def compute_features(image, feature_type):
    r"""
    Computes a particular feature representation of the given images.

    Parameters
    ----------
    image: :class:`pybug.image.MaskedNDImage`
        The original image from which the features will be computed.
    feature_type: string or closure
        If None, no feature representation will be computed from the
        original image.
        If string or closure, the feature representation will be computed
        in the following way:
            If string, the feature representation will be extracted by
            executing:

                feature_image = eval('img.features.' + feature_type + '()')

            For this to work properly feature_type needs to be one of
            Pybug's standard image feature methods. Note that, in this case,
            the feature computation will be carried out using its default
            options.

            Non-default feature options and new experimental feature can be
            used by defining a closure. In this case, the closure must define a
            function that receives as an input an image and returns a
            particular feature representation of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `pybug.image.feature.py` for details more details on
            Pybug's standard image features and feature options.

    Returns
    -------
    feature_image: :class:`pybug.image.MaskedNDImage`
        The resulting feature image.
    """
    if feature_type is not None:
        if type(feature_type) is str:
            image = eval('image.features.' + feature_type + '()')
        elif hasattr(feature_type, '__call__'):
            image = feature_type(image)
        else:
            raise ValueError("feature_type can only be: (1) None, "
                             "(2) a string defining one of Pybug's standard "
                             "image feature_type ('hog', 'igo', etc) "
                             "or (3) a closure defining a non-standard "
                             "feature computation")
    return image


def mean_pointcloud(pointcloud_list):
    r"""
    Compute the mean of a list of point cloud objects

    Parameters
    ----------
    pointcloud_list: list of :class:`pybug.shape.PointCloud`
        List of point cloud objects from which we want to
        compute the mean.

    Returns
    -------
    mean_pointcloud: class:`pybug.shape.PointCloud`
        The mean point cloud.
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
