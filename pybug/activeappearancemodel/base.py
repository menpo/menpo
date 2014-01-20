from __future__ import division
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from pybug.shape import TriMesh
from pybug.transform .affine import Scale, SimilarityTransform, UniformScale, \
    AffineTransform, Translation
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.transform.modeldriven import OrthoMDTransform
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.transform.tps import TPS
from pybug.model import PCAModel
from pybug.lucaskanade.residual import LSIntensity
from pybug.lucaskanade.appearance import AlternatingInverseCompositional
from pybug.activeappearancemodel.functions import mean_pointcloud,\
    build_reference_frame, build_patch_reference_frame, compute_features,\
    compute_error_facesize, compute_error_me17, compute_error_p2p, \
    compute_error_rms


class AAM(object):
    r"""
    An Active Appearance Model (AAM)


    Parameters
    -----------
    shape_model:
    reference_frame:
    appearance_model_pyramid:
    """

    def __init__(self, shape_model_pyramid, appearance_model_pyramid,
                 features_dic, downscale):
        self.shape_model_pyramid = shape_model_pyramid
        self.appearance_model_pyramid = appearance_model_pyramid
        self.features_dic = features_dic
        self.downscale = downscale

    @property
    def n_levels(self):
        return len(self.appearance_model_pyramid)

    def _feature_pyramid(self, image, reference_landmarks):
        r"""
        Compute to an image using Lucas-Kanade.

        Parameters
        -----------
        image: :class:`pybug.image.IntensityImage`
            The image to which the

        Returns
        -------
        feature_pyramid:
        """
        image = image.rescale_to_reference_landmarks(reference_landmarks)
        pyramid = image.gaussian_pyramid(n_levels=self.n_levels)
        image_pyramid = [compute_features(p.next(), self.features_dic)
                         for p in pyramid]
        return image_pyramid.reverse()

    def instance(self, shape_weights, appearance_weights, level=-1,
                 transform_cls=PiecewiseAffineTransform, patches=False,
                 patch_size=(16, 16), interpolator='scipy', **warp_kwargs):
        r"""
        Create a novel AAM instance using the given shape and appearance
        weights.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or list
            Weights of the shape model that should be used to create
            a novel shape instance.

        appearance_weights: (n_weights,) ndarray or list
            Weights of the appearance model that should be used to create
            a novel appearance instance.

        level: int, optional
            Index representing the appearance model to be used to create
            the instance.

            Default: -1
        transform_cls: :class:`pybug.transform.base.PureAlignmentTransform`
            Class of transform that should be used to warp the appearance
            instance onto the reference frame defined by the shape instance.

            Default: PieceWiseAffineTransform
        interpolator: 'scipy' or 'c', optional
            The interpolator that should be used to perform the warp.

            Default: 'scipy'
        kwargs: dict
            Passed through to the interpolator. See `pybug.interpolation`
            for details.

        Returns
        -------
        cropped_image: :class:`pybug.image.masked.MaskedNDImage`
            The novel AAM instance.
        """
        if patches:
            transform_cls = TPS

        template = self.appearance_model_pyramid[level].mean
        landmarks = template.landmarks['source'].lms

        # multiply weights by std
        n_shape_weights = len(shape_weights)
        shape_weights *= self.shape_model_pyramid[level].eigenvalues[
            :n_shape_weights] ** 0.5
        # compute shape instance
        shape_instance = \
            self.shape_model_pyramid[level].instance(shape_weights)
        # build reference frame using to shape instance
        if patches:
            reference_frame = build_patch_reference_frame(
                shape_instance, patch_size=patch_size)
        else:
            if type(landmarks) == TriMesh:
                trilist = template.landmarks['source'].lms.trilist
            else:
                trilist = None
            reference_frame = build_reference_frame(shape_instance,
                                                    trilist=trilist)

        # select appearance model
        n_appearance_weights = len(appearance_weights)
        appearance_model = self.appearance_model_pyramid[level]
        # multiply weights by std
        appearance_weights *= appearance_model.eigenvalues[
            :n_appearance_weights] ** 0.5
        # compute appearance instance
        appearance_instance = appearance_model.instance(appearance_weights)

        # compute transform mapping appearance instance to previous
        # reference frame
        transform = transform_cls(reference_frame.landmarks['source'].lms,
                                  landmarks)
        # return warped appearance
        return appearance_instance.warp_to(reference_frame.mask, transform,
                                           interpolator, **warp_kwargs)

    def initialize_lk(self, md_transform_cls=OrthoMDTransform,
                      transform_cls=PiecewiseAffineTransform,
                      global_transform_cls=SimilarityTransform,
                      lk_algorithm=AlternatingInverseCompositional,
                      residual_dic={'type': LSIntensity, 'options': None},
                      n_shape=None, n_appearance=None):
        r"""
        Initialize a particular Lucas-Kanade algorithm for fitting this AAM
        onto images.

        Parameters
        -----------
        md_transform_cls: :class:`pybug.transform.base.modeldriven`

            Default: OrthoMDTransform
        transform_cls: :class:`pybug.transform.base.PureAlignmentTransform`

            Default: PieceWiseAffineTransform
        global_transform_cls: :class:`pybug.transform.base.affine`

            Default: SimilarityTransform
        lk_algorithm: :class:`pybug.lucakanade.appearance`

            Default: PieceWiseAffineTransform
        residual: :class:`pybug.lucakanade.residual`

            Default: 'LSIntensity'
        """
        self._lk_pyramid = []

        for am, sm, n_s, n_a in zip(self.appearance_model_pyramid,
                                    self.shape_model_pyramid,
                                    n_shape, n_appearance):

            global_transform = global_transform_cls(np.eye(3, 3))
            source = am.mean.landmarks['source'].lms

            if n_shape is not None:
                sm.n_active_components = n_s
            md_transform = md_transform_cls(
                sm, transform_cls, global_transform, source=source)

            if n_appearance is not None:
                am.n_active_components = n_a

            if residual_dic['options'] is None:
                residual = residual_dic['type']()
            else:
                residual = residual_dic['type'](**residual_dic['options'])

            self._lk_pyramid.append(lk_algorithm(am, residual,
                                                 md_transform))

    def _lk_fit(self, image_pyramid, initial_landmarks, max_iters=20):
        r"""
        Fit the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image_pyramid:
        initial_landmarks: :class:`pybug.shape.PointCloud`
        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        Returns
        -------
        md_tranform_pyramid:
            A list containing the optimal transform per pyramidal level.
        """

        target = initial_landmarks
        md_transform_pyramid = []

        for i, lk in zip(image_pyramid, self._lk_pyramid):
            lk.transform.target = target
            md_transform = lk.align(i, lk.transform.as_vector(),
                                    max_iters=max_iters)
            md_transform_pyramid.append(md_transform)
            target = Scale(self.downscale, n_dims=md_transform.n_dims).apply(
                md_transform.target)

        return md_transform_pyramid

    def lk_fit_landmarked_image(self, image, group=None, label='all',
                                runs=10, noise_std=0.05, max_iters=20,
                                verbose=True, view=False):
        r"""
        Fit the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image:
        noise_std: :class:`pybug.shape.PointCloud`
        group:
        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        Returns
        -------
        md_transform_pyramid:
        """

        scale = self.downscale ** (self.n_levels-1)

        image = image.rescale_to_reference_landmarks(
            self.shape_model_pyramid[-1].mean)
        pyramid = image.gaussian_pyramid(n_levels=self.n_levels,
                                         downscale=self.downscale)
        image_pyramid = [compute_features(p, self.features_dic)
                         for p in pyramid]
        image_pyramid.reverse()

        original_landmarks = image.landmarks[group].lms

        affine_correction = AffineTransform.align(
            image_pyramid[-1].landmarks[group].lms, original_landmarks)

        optimal_transforms = []
        for j in range(runs):
            reference_landmarks = self.shape_model_pyramid[0].mean
            transform = noisy_align(reference_landmarks,
                                    image_pyramid[0].landmarks[group].lms,
                                    noise_std=noise_std)
            initial_landmarks = transform.apply(reference_landmarks)
            optimal_transforms.append(self._lk_fit(
                image_pyramid, initial_landmarks, max_iters=max_iters))

            fitted_landmarks = optimal_transforms[j][-1].target
            fitted_landmarks = affine_correction.apply(fitted_landmarks)

            if verbose:
                error = compute_error_facesize(fitted_landmarks.points,
                                               original_landmarks.points)
                print ' - run {} of {} with error: {}'.format(j+1, runs,
                                                              error)

            # ToDo: This will need to change with the new LK structure
            image.landmarks['initial_{}'.format(j)] = \
                affine_correction.apply(UniformScale(scale, 2).apply(
                    initial_landmarks))
            image.landmarks['fitted_{}'.format(j)] = fitted_landmarks
            if view:
                image.landmarks['initial_{}'.format(j)].view(
                    include_labels=False)
                image.landmarks['fitted_{}'.format(j)].view(
                    include_labels=False)
                plt.show()

        return optimal_transforms

    def lk_fit_landmarked_database(self, images, group=None, label='all',
                                   runs=10, noise_std=0.5, max_iters=20,
                                   verbose=True, view=False):
        r"""
        Fit the AAM to a list of landmark images using Lukas-Kanade.

        Parameters
        -----------
        images:
        runs:
        noise_std:
        group:
        max_iters:

        Returns
        -------
        optimal_transforms:
        """
        n_images = len(images)
        optimal_transforms = []
        for j, i in enumerate(images):
            if verbose:
                print '- fitting image {} of {}'.format(j+1, n_images)
            optimal_transforms.append(self.lk_fit_landmarked_image(
                i, runs=runs, label=label, noise_std=noise_std, group=group,
                max_iters=max_iters, verbose=verbose, view=view))

        return optimal_transforms


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


def aam_builder(images, group=None, label='all', interpolator='scipy',
                reference_shape=None, scale=1,
                boundary=3, trilist=None,
                patches=False, patch_size=(16, 16), n_levels=3,
                scaled_reference_frames=False, downscale=2,
                transform_cls=PiecewiseAffineTransform,
                features_dic=None,
                max_shape_components=None, max_appearance_components=None):

    r"""

    Parameters
    ----------
    images: list
        The images from which to build the Active Appearance Model
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
    trilist: (Nt, 3) ndarray, optional

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

    if reference_shape is None:
        print '- Computing reference shape'
        shapes = [i.landmarks[group][label].lms for i in images]
        reference_shape = mean_pointcloud(shapes)

    print '- Rescaling images to reference shape'
    images = [i.rescale_to_reference_landmarks(reference_shape,
                                               group=group, label=label,
                                               interpolator=interpolator)
              for i in images]

    print '- Setting gaussian pyramid generators'
    pyramids = [i.gaussian_pyramid(n_levels=n_levels, downscale=downscale)
                for i in images]

    print '- Computing features'
    images = [compute_features(p.next(), features_dic)
              for p in pyramids]
    # extract potentially rescaled shapes
    shapes = [i.landmarks[group][label].lms for i in images]

    print '- Building shape model'
    # centralize shapes
    centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
    # align centralized shape using Procrustes Analysis
    gpa = GeneralizedProcrustesAnalysis(centered_shapes)
    aligned_shapes = [s.aligned_source for s in gpa.transforms]
    # scale shapes if necessary
    if scale != 1:
        aligned_shapes = [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                          for s in aligned_shapes]

    # build shape model
    shape_model = PCAModel(aligned_shapes)
    if max_shape_components is not None:
        # trim shape model if required
        shape_model.trim_components(max_shape_components)

    print '- Building reference frame'
    reference_shape = mean_pointcloud(aligned_shapes)
    if patches:
        # build patch based reference frame
        reference_frame = build_patch_reference_frame(
            reference_shape, boundary=boundary, patch_size=patch_size)
    else:
        # build reference frame
        reference_frame = build_reference_frame(
            reference_shape, boundary=boundary, trilist=trilist)

    print '- Building model pyramids'
    shape_model_pyramid = []
    appearance_model_pyramid = []
    # for each level
    for j in range(n_levels):
        print ' - Level {}'.format(j)

        if j != 0:
            print ' - Computing features'
            images = [compute_features(p.next(), features_dic)
                      for p in pyramids]

            print ' - Building shape model'
            if scaled_reference_frames:
                shapes = [i.landmarks[group][label].lms for i in images]
                centered_shapes = [Translation(-s.centre).apply(s)
                                   for s in shapes]
                gpa = GeneralizedProcrustesAnalysis(centered_shapes)
                aligned_shapes = [s.aligned_source for s in gpa.transforms]
                if scale != 1:
                    aligned_shapes = \
                        [Scale(scale, n_dims=reference_shape.n_dims).apply(s)
                         for s in aligned_shapes]
                # build shape model
                shape_model = PCAModel(aligned_shapes)
                if max_shape_components is not None:
                    # trim shape model if required
                    shape_model.trim_components(max_shape_components)

                print ' - Building reference frame'
                reference_shape = mean_pointcloud(aligned_shapes)
                if patches:
                    # build patch based reference frame
                    reference_frame = build_patch_reference_frame(
                        reference_shape, boundary=boundary, patch_size=patch_size)
                else:
                    # build reference frame
                    reference_frame = build_reference_frame(
                        reference_shape, boundary=boundary, trilist=trilist)

        # add shape model to the list
        shape_model_pyramid.append(shape_model)

        print ' - Computing transforms'
        transforms = [transform_cls(reference_frame.landmarks['source'].lms,
                                    i.landmarks[group][label].lms)
                      for i in images]

        print ' - Warping images'
        images = [i.warp_to(reference_frame.mask, t,
                            interpolator=interpolator)
                  for i, t in zip(images, transforms)]

        for i in images:
            i.landmarks['source'] = reference_frame.landmarks['source']
        if patches:
            for i in images:
                i.build_mask_around_landmarks(patch_size, group=group,
                                              label=label)
        else:
            for i in images:
                i.constrain_mask_to_landmarks(group='source', trilist=trilist)

        print ' - Building appearance model'
        appearance_model = PCAModel(images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)

        # add appearance model to the list
        appearance_model_pyramid.append(appearance_model)

    # reverse the list of shape and appearance models so that they are
    # ordered from lower to higher resolution
    shape_model_pyramid.reverse()
    appearance_model_pyramid.reverse()

    return AAM(shape_model_pyramid, appearance_model_pyramid,
               features_dic, downscale)
