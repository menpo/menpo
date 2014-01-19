from __future__ import division
import matplotlib.pylab as plt
import numpy as np
from pybug.shape import PointCloud, TriMesh
from pybug.landmark.labels import labeller
from pybug.transform .affine import Translation
from pybug.image import MaskedNDImage


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
    reference_frame = MaskedNDImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame


def compute_features(image, features_dic):
    r"""
    Compute feature images.

    Parameters
    ----------
    image: :class:`pybug.image.MaskedNDImage`
        The original image from which the features will be computed
    features_dic: dic
        ['type'] : string
            String specifying the type of features to compute
        ['options']: **kwargs:
            Passed through to the particular feature method being used. See
            `pybug/image/MaskedNDImage` for details on feature options.

    Returns
    -------
    feature_image: :class:`pybug.image.MaskedNDImage`
        The resulting feature image.
    """
    if features_dic is not None:
        if features_dic['type'] is 'normalize_std':
            image.normalize_std_inplace(**features_dic['options'])
        elif features_dic['type'] is 'normalize_norm':
            image.normalize_norm_inplace(**features_dic['options'])
        elif features_dic['type'] is 'euler':
            raise NotImplementedError("Euler features not implemented yet")
        elif features_dic['type'] is 'igo':
            raise NotImplementedError("IGO features not implemented yet")
        elif features_dic['type'] is 'es':
            raise NotImplementedError("ES features not implemented yet")
        elif features_dic['type'] is 'hog':
            raise NotImplementedError("HoG features not implemented yet")
        elif features_dic['type'] is 'sift':
            raise NotImplementedError("Sift features not implemented yet")

    # TODO: These should disappear with the new image refactoring
    nd_image = MaskedNDImage(image.pixels, mask=image.mask)
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


def compute_error_rms(fitted, ground_truth):
    return np.sqrt(np.mean((fitted.flatten() - ground_truth.flatten()) ** 2))


def compute_error_p2p(fitted, ground_truth):
    return np.mean(np.sqrt(np.sum((fitted - ground_truth) ** 2, axis=-1)))


def compute_error_facesize(fitted, ground_truth):
    face_size = np.mean(np.max(ground_truth, axis=0) -
                        np.min(ground_truth, axis=0))
    return compute_error_p2p(fitted, ground_truth) / face_size


def compute_error_me17(fitted, ground_truth, leye, reye):
    return (compute_error_p2p(fitted, ground_truth) /
            compute_error_p2p(leye, reye))


def plot_ced(fitted_shapes, original_shape, error_type='face_size',
             label=None):

    if error_type is 'rms':
        errors = [compute_error_rms(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001
    elif error_type is 'p2p':
        errors = [compute_error_p2p(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001
    elif error_type is 'face_size':
        errors = [compute_error_facesize(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001

        plt.xlabel('point-to-point error normalized by face size')
        plt.ylabel('proportion of images')

    elif error_type is 'me17':
        errors = [compute_error_me17(f.points, o.points)
                  for f, o in zip(fitted_shapes, original_shape)]
        stop = 0.1
        step = 0.001

    n_shapes = len(fitted_shapes)
    error_axis = np.arange(0, stop, step)
    proportion_axis = [np.count_nonzero(errors < limit) / n_shapes
                       for limit in error_axis]

    error_median = np.median(errors)
    error_mean = np.mean(errors)
    error_std = np.std(errors)

    text = label + '  median = {}  mean = {}  std = {}'.format(
        error_median, error_mean, error_std)

    plt.plot(error_axis, proportion_axis, label=text)

    plt.title('Cumulative Error Distribution')
    plt.grid(True)
    plt.legend()
    plt.show()

    return errors, error_median, error_mean, error_std