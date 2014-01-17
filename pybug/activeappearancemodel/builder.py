from __future__ import division
import numpy as np
from skimage.transform import pyramid_gaussian
from pybug.shape import PointCloud
from pybug.landmark.labels import labeller, ibug_68_trimesh
from pybug.transform import Scale, Translation, SimilarityTransform
from pybug.transform.affine import UniformScale
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.image import MaskedNDImage, RGBImage, BooleanNDImage
from pybug.model import PCAModel


def build_reference_frame(reference_landmarks, scale=1, boundary=3,
                          group='source', triangulation=None):
    r"""
    Build reference frame from reference landmarks.

    Parameters
    ----------
    reference_landmarks:
    scale: int, optional

        Default: 1
    boundary: int, optional

        Default: 3
    group: str, optional

        Default: 'source'
    triangulation: dictionary, optional

        Default: None

    Returns
    -------
    rescaled_image : type(self)
        A copy of this image, rescaled.
    """
    # scale reference shape if necessary
    if scale is not 1:
        n_dims = reference_landmarks.n_dims
        reference_landmarks = Scale(
            scale, n_dims=n_dims).apply(reference_landmarks)

    # compute lower bound
    lower_bound = reference_landmarks.bounds(boundary=boundary)[0]
    # translate reference shape using lower bound
    reference_landmarks = Translation(-lower_bound).apply(reference_landmarks)

    # compute reference frame resolution
    reference_resolution = reference_landmarks.range(boundary=boundary)

    # build reference frame
    reference_frame = MaskedNDImage.blank(reference_resolution)

    # assign landmarks
    reference_frame.landmarks[group] = reference_landmarks
    # check for precomputed triangulation
    if triangulation is not None:
        labeller([reference_frame], group, triangulation['function'])
        trilist = \
            reference_frame.landmarks[triangulation['label']].lms.trilist
    else:
        trilist = None
    # mask reference frame
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame, trilist


def build_patch_reference_frame(reference_landmarks, scale=1,
                                patch_size=[16, 16], group='source'):
    r"""
    Build reference frame from reference landmarks.

    Parameters
    ----------
    reference_landmarks:
    scale: int, optional

        Default: 1
    boundary: int, optional

        Default: 3
    group: str, optional

        Default: 'source'
    triangulation: dictionary, optional

        Default: None

    Returns
    -------
    rescaled_image : type(self)
        A copy of this image, rescaled.
    """
    boundary = np.max(patch_size)
    # scale reference shape if necessary
    if scale is not 1:
        n_dims = reference_landmarks.n_dims
        reference_landmarks = Scale(
            scale, n_dims=n_dims).apply(reference_landmarks)

    # compute lower bound
    lower_bound = reference_landmarks.bounds(boundary=boundary)[0]
    # translate reference shape using lower bound
    reference_landmarks = Translation(-lower_bound).apply(reference_landmarks)

    # compute reference frame resolution
    reference_resolution = reference_landmarks.range(boundary=boundary)

    # build reference frame
    reference_frame = MaskedNDImage.blank(reference_resolution)

    # assign landmarks
    reference_frame.landmarks[group] = reference_landmarks

    # mask reference frame
    mask = build_patch_mask(reference_frame, reference_landmarks,
                            patch_size=patch_size)
    reference_frame.mask = BooleanNDImage(mask)

    return reference_frame


# TODO: Should this be a method in AbstractNDImage?
def rescale_to_reference_landmarks(image, reference_landmarks,
                                   scale=1, group=None, label=None,
                                   interpolator='scipy', **kwargs):
    r"""
    Return a copy of this image, rescaled so that the scale of a
    particular group of landmarks matches the scale of the given
    reference landmarks.

    Parameters
    ----------
    image:
    reference_landmarks:
    scale: int, optional

        Default: 1
    group: str, optional

        Default: None
    label: str, optional

        Default: None
    interpolator: str, optional

        Default: 'scipy'
    **kwargs: optional

    Returns
    -------
    rescaled_image : type(self)
        A copy of this image, rescaled.
    """
    # scale reference shape if necessary
    if scale is not 1:
        n_dims = reference_landmarks.n_dims
        reference_landmarks = Scale(
            scale, n_dims=n_dims).apply(reference_landmarks)

    # obtain object's shape
    current_shape = image.landmarks[group][label].lms
    # compute scale difference between current object shape and reference
    # shape
    scale = UniformScale.align(current_shape, reference_landmarks).as_vector()
    # return rescale image
    return image.rescale(scale, interpolator=interpolator, **kwargs)


# TODO: Should this be a method in MaskedNDImage?
def build_patch_mask(image, shape, patch_size):

    mask = np.zeros(image.shape)
    patch_half_size = np.asarray(patch_size) / 2

    for i, point in enumerate(shape.points):
        start = np.floor(point - patch_half_size).astype(int)
        finish = np.floor(point + patch_half_size).astype(int)
        x, y = np.mgrid[start[0]:finish[0], start[1]:finish[1]]

        # deal with boundaries
        x[x > image.shape[0] - 1] = image.shape[0] - 1
        y[y > image.shape[1] - 1] = image.shape[1] - 1
        x[x < 0] = 0
        y[y < 0] = 0

        mask[x.flatten(), y.flatten()] = True

    return mask


# TODO: Should this be a method in AbstractNDImage?
def gaussian_pyramid(image, n_levels=None, downscale=2, sigma=None, order=1,
                     mode='reflect', cval=0):
    r"""

    Parameters
    ----------
    image:
    n_levels:
    downscale:
    sigma:
    order:
    mode:
    cval:

    Returns
    -------
    image_pyramid:
    """

    if n_levels is None:
        max_layer = -1
    else:
        max_layer = n_levels - 1

    image_iterator = pyramid_gaussian(
        image.pixels, max_layer=max_layer, downscale=downscale, sigma=sigma,
        order=order, mode=mode, cval=cval)

    mask_iterator = pyramid_gaussian(
        image.mask.pixels, max_layer=max_layer, downscale=downscale,
        sigma=0, order=order, mode=mode, cval=cval)

    # TODO: Bug!!!
    # the current .squeeze() on image_data does not work properly when
    # max_layer=-1 for IntensityImages, due to the special way in which this
    # image class is constructed (removing the channel axis).
    pyramid = [image.__class__(image_data.squeeze(axis=(2,)),
                               mask=mask_data.squeeze(axis=(2,)))
               for image_data, mask_data in zip(image_iterator,
                                                mask_iterator)]
    # rescale and reassign landmarks if necessary
    for j, i, in enumerate(pyramid):
        i.landmarks = image.landmarks
        transform = UniformScale(downscale ** j, 2)
        transform.pseudoinverse.apply_inplace(i.landmarks)

    return pyramid


# TODO: Should this be a method on SimilarityTransform?
# and in Transforms in general?
def noisy_align(source, target, noise_std=0.05, rotation=False):
    r"""

    Parameters
    ----------
    source :
    target :
    noise_std:
    rotation:

    Returns
    -------
    noisy_transform :
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


def compute_features(image, feature_type, **kwargs):
    r"""

    Parameters
    ----------
    image:
    feature_type :
    **kwargs:

    Returns
    -------
    feature_image :
    """
    if feature_type is 'norm':
        image.normalize_inplace(**kwargs)
    elif feature_type is 'igo':
        image = image.features.igo(**kwargs)
    elif feature_type is 'hog':
        image = image.features.hog(**kwargs)

    nd_image = MaskedNDImage(image.pixels, mask=image.mask)
    nd_image.landmarks = image.landmarks

    return nd_image


def compute_mean_pointcloud(pointcloud_list):
    r"""

    Parameters
    ----------
    pointcloud_list:

    Returns
    -------
    mean_pointcloud :
    """
    return PointCloud(np.mean([pc.points for pc in pointcloud_list], axis=0))


def aam_builder(images, group='PTS', label='all', interpolator='scipy',
                reference_landmarks=None, scale=1, crop_boundary=0.2,
                reference_frame_boundary=3, triangulation=None,
                patches=False, patch_size=[16, 16], n_levels=3,
                transform_cls=PiecewiseAffineTransform,
                features={'type': None, 'options': None},
                max_shape_components=None, max_appearance_components=None):

    r"""

    Parameters
    ----------
    images:
    group: str, optional

        Default: 'PTS'
    label: tr, optional

        Default: 'all'
    interpolator: str, optional

        Default: 'scipy'
    reference_landmarks: optional

        Default: None
    scale: float, optional

        Default: 1
    crop_boundary: float, optional

        Default: 0.2
    reference_frame_boundary: int, optional

        Default: 3
    triangulation: dictionary, optional

        Default: None
    n_levels: int, optional

        Default: 3
    transform_cls: optional

        Default: PieceWiseAffine
    feature_space: dictionary, optional

        Default: None
    max_shape_components: float, optional

        Default: 0.95
    max_appearance_components: float, optional

        Deafult: 0.95


    Returns
    -------
    aam : :class:`pybug.activeappearancemodel.AAM`
    """

    # TODO:
    print '- Cropping images'
    # crop images around their landmarks
    for i in images:
        i.crop_to_landmarks_proportion(crop_boundary, group=group, label=label)

    # TODO:
    if reference_landmarks is None:
        print '- Compute reference shape'
        # extract original shapes
        shapes = [i.landmarks[group][label].lms for i in images]
        # define reference shape
        reference_shape = compute_mean_pointcloud(shapes)

    # TODO:
    print '- Rescaling images to reference shape'
    # rescale images so that the scale of their corresponding shapes matches
    # the scale of the reference shape
    images = [rescale_to_reference_landmarks(i, reference_shape, group=group,
                                             label=label,
                                             interpolator=interpolator)
              for i in images]

    # TODO:
    print '- Building gaussian pyramid'
    # build gaussian pyramids
    images_pyramid = [gaussian_pyramid(i, n_levels=n_levels) for i in images]

    # TODO:
    print '- Computing features'
    # compute features
    images_pyramid = [[compute_features(i, features['type'],
                                        **features['options'])
                       for i in images]
                      for images in images_pyramid]

    # extract rescaled shapes
    shapes = [i[0].landmarks[group][label].lms for i in images_pyramid]

    # TODO:
    print '- Building shape model'
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]
    # scale shapes if necessary
    if scale is not 1:
        aligned_shapes = [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                          for s in aligned_shapes]
    # build shape model
    shape_model = PCAModel(aligned_shapes)
    # trim shape model if required
    if max_shape_components is not None:
        shape_model.trim_components(max_shape_components)

    # TODO:
    print '- Building reference frame'
    if patches:
        # build reference frame
        reference_frame = build_patch_reference_frame(
            reference_shape, scale=scale, patch_size=patch_size)
        # mask images
        for i, s in zip(images, shapes):
                mask = build_patch_mask(i, s, patch_size=patch_size)
                i.mask = BooleanNDImage(mask)

    else:
        # build reference frame
        reference_frame, trilist = build_reference_frame(
            reference_shape, scale=scale, boundary=reference_frame_boundary,
            triangulation=triangulation)
        # mask images
        for i in images:
                i.constrain_mask_to_landmarks(group=group, trilist=trilist)

    # free memory
    del images, aligned_shapes, gpa, centered_shapes, shapes

    # TODO:
    print '- Building appearance models'
    # initialize list of appearance models
    appearance_model_pyramid = []
    # for each level
    for j in range(n_levels):
        print ' - Level {}'.format(j)
        # obtain level images
        images_level = [p[j] for p in images_pyramid]

        # compute transforms
        transforms = [transform_cls(reference_frame.landmarks['source'].lms,
                                    i.landmarks[group].lms)
                      for i in images_level]
        # warp images
        images = [i.warp_to(reference_frame.mask, t,
                            interpolator=interpolator)
                  for i, t in zip(images_level, transforms)]
        # assign landmarks using default group and label
        for i in images_level:
            i.landmarks[group] = reference_frame.landmarks['source']
        # mask images
        if patches:
            # patch mask
            for i in images:
                    mask = build_patch_mask(
                        i, reference_frame.landmarks['source'].lms,
                        patch_size=patch_size)
                    i.mask = BooleanNDImage(mask)
        else:
            # convex hull mask
            for i in images_level:
                i.constrain_mask_to_landmarks(group=group, trilist=trilist)
        # build appearance model
        appearance_model = PCAModel(images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)
        # add appearance model to the list
        appearance_model_pyramid.append(appearance_model)
    # reverse the list of appearance models
    appearance_model_pyramid.reverse()
    from pybug.activeappearancemodel.base import AAM
    return AAM(shape_model, reference_frame, appearance_model_pyramid,
               features)
