from __future__ import division
import abc
import numpy as np
from copy import deepcopy
from pybug.landmark import LandmarkGroup
from pybug.transform .affine import \
    Scale, SimilarityTransform, AffineTransform
from pybug.transform.modeldriven import OrthoMDTransform, ModelDrivenTransform
from pybug.lucaskanade.residual import LSIntensity
from pybug.lucaskanade.appearance import AlternatingInverseCompositional
from pybug.aam.base import AAM
from pybug.aam.functions import noisy_align, compute_features
from pybug.aam.fitting import FittingList, AAMFitting


class Fitter(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _set_up(self, **kwargs):
        r"""
        Sets up the Fitter object.
        """
        pass

    @abc.abstractproperty
    def _source(self):
        pass

    @abc.abstractmethod
    def _build_fitting(self, image, basic_fittings, affine_correction,
                       ground_truth=None, error_type='me_norm'):
        pass

    @abc.abstractmethod
    def _prepare_image(self, image, initial_target, gt_target=None):
        pass

    @abc.abstractmethod
    def fit(self, image_list, initial_target, gt_target_list=None,
            max_iters=20, **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image_list: :class:`pybug.image.masked.MaskedImage` list
            A list containing the images to be fitted.

        initial_target: :class:`pybug.shape.PointCloud`
            The initial target from which the fitting will start.

        gt_target_list: :class:`pybug.shape.PointCloud` list,
                                  optional
            A list containing the ground truth targets for each images in
            image_list.

            Default: None

        max_iters:

        Returns
        -------
        fittings: :class:`pybug.aam.fitting` list
            The list of obtained fitting objects associated to the images in
            image_list.
        """
        pass

    def _target_detection(self, noise_std=0.0, rotation=False):
        raise ValueError('_taget_detection not implemented yet')

    def _noisy_align_from_gt_target(self, gt_target, noise_std=0.0,
                                    rotation=False):
        source = self._source
        return noisy_align(source, gt_target, noise_std=noise_std,
                           rotation=rotation).apply(source)

    def fit_image(self, image, runs=5, noise_std=0.0, rotation=False,
                  max_iters=20, verbose=True, view=False, **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The landmarked image to be fitted.

        noise_std: float
            The standard deviation of the white noise used to perturb the
            landmarks.

            Default: 0.05

        rotation: boolean
            If False the second parameter of the SimilarityTransform,
            which captures captures in-plane rotations, is set to 0.

            Default:False

        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        verbose: boolean
            If True the error between the ground truth landmarks and the
            result of the fitting is displayed.

            Default: True

        view: boolean
            If True the final result of the fitting procedure is shown.

            Default: False

        Returns
        -------
        optimal_transforms: list of
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level.
        """
        image = deepcopy(image)
        initial_target = self._target_detection(noise_std=noise_std,
                                                rotation=rotation)
        fitting_list = []
        for _ in range(runs):
            fitting = self._fit_image(image, initial_target,
                                      max_iters=max_iters, gt_target=None,
                                      verbose=verbose, view=view, **kwargs)
            fitting_list.append(fitting)

        return FittingList(fitting_list)

    def fit_landmarked_image(self, image, group=None, label='all',
                             initialization='from_gt_target', runs=5,
                             noise_std=0.0, rotation=False, max_iters=20,
                             verbose=True, view=False, **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The landmarked image to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        initialization: string, optional

        noise_std: float
            The standard deviation of the white noise used to perturb the
            detection.

            Default: 0.0

        rotation: boolean
            If False, the chosen initialization method will discard
            possible rotation information.

            Default:False

        max_iters: int, optional
            The number of iterations per level

            Default: 20

        verbose: boolean

            Default: True

        view: boolean

            Default: False

        Returns
        -------
        fitting_lis: list of
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level.
        """
        image = deepcopy(image)
        gt_target = image.landmarks[group][label].lms

        fitting_list = []
        for _ in range(runs):

            if initialization is 'from_gt_target':
                initial_target = self._noisy_align_from_gt_target(
                    gt_target, noise_std=noise_std, rotation=rotation)
            elif type is 'detection':
                initial_target = self._target_detection(
                    noise_std=noise_std, rotation=rotation)
            else:
                raise ValueError('Unknown initialization string selected. Valid'
                                 'options are: from_gt_target, detection')

            fitting = self._fit_image(image, initial_target,
                                      max_iters=max_iters, gt_target=gt_target,
                                      verbose=verbose, view=view, **kwargs)
            fitting_list.append(fitting)

        return FittingList(fitting_list)

    def _fit_image(self, image, initial_target, max_iters=20,
                   gt_target=None, verbose=True, view=False, **kwargs):
        image_list = self._prepare_image(image, initial_target,
                                         gt_target=gt_target)

        if gt_target:
            gt_target_list = [i.landmarks['gt_target'].lms
                              for i in image_list]
        else:
            gt_target_list = None

        initial_target_list = [i.landmarks['initial_target'].lms
                               for i in image_list]

        affine_correction = AffineTransform.align(initial_target_list[-1],
                                                  initial_target)

        basic_fitting_list = self.fit(image_list, initial_target_list[0],
                                      max_iters=max_iters,
                                      gt_target_list=gt_target_list, **kwargs)

        fitting = self._build_fitting(image, basic_fitting_list,
                                      affine_correction,
                                      ground_truth=gt_target)
        if verbose:
            # fitting.print_final_cost
            if gt_target:
                fitting.print_final_error()
        if view:
            fitting.view_final_target(new_figure=True)

        return fitting

    def fit_image_list(self, images, runs=5, noise_std=0.0, rotation=False,
                       max_iters=20, verbose=True, view=False, **kwargs):
        r"""
        Fits the AAM to a list of landmark images using Lukas-Kanade.

        Parameters
        -----------
        images: list of :class:`pybug.image.IntensityImage`
            The list of landmarked images to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        noise_std: float
            The standard deviation of the white noise used to perturb the
            landmarks.

            Default: 0.05

        rotation: boolean
            If False the second parameter of the SimilarityTransform,
            which captures captures inplane rotations, is set to 0.

            Default:False

        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        verbose: boolean
            If True the error between the ground truth landmarks and the
            result of the fitting is displayed.

            Default: True

        view: boolean
            If True the final result of the fitting procedure is shown.

            Default: False

        Returns
        -------
        optimal_transforms: list of list
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level for
            all images.
        """
        new_kwargs = {'runs': runs, 'noise_std': noise_std,
                      'rotation': rotation, 'max_iters': max_iters,
                      'verbose': verbose, 'view': view, 'kwargs': kwargs}
        kwargs = dict(new_kwargs.items() + kwargs.items())

        self._fitting_function = self.fit_image

        return self._fit_image_list(images, **kwargs)

    def fit_landmarked_image_list(self, images, group=None, label='all',
                                  initialization='from_gt_target', runs=5,
                                  noise_std=0.0, rotation=False, max_iters=20,
                                  verbose=True, view=False, **kwargs):
        r"""
        Fits the AAM to a list of landmark images using Lukas-Kanade.

        Parameters
        -----------
        images: list of :class:`pybug.image.IntensityImage`
            The list of landmarked images to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        noise_std: float
            The standard deviation of the white noise used to perturb the
            landmarks.

            Default: 0.05

        rotation: boolean
            If False the second parameter of the SimilarityTransform,
            which captures captures inplane rotations, is set to 0.

            Default:False

        max_iters: int, optional
            The number of iterations per pyramidal level

            Default: 20

        verbose: boolean
            If True the error between the ground truth landmarks and the
            result of the fitting is displayed.

            Default: True

        view: boolean
            If True the final result of the fitting procedure is shown.

            Default: False

        Returns
        -------
        optimal_transforms: list of list
                             :class:`pybug.transform.ModelDrivenTransform`
            A list containing the optimal transform per pyramidal level for
            all images.
        """
        if not isinstance(images[0].landmarks[group][label], LandmarkGroup):
            raise ValueError('All images must be labelled with '
                             'the specified group ({}) and labels '
                             '({})'.format(group, label))

        new_kwargs = {'group': group, 'label': label,
                      'initialization': initialization, 'runs': runs,
                      'noise_std': noise_std, 'rotation': rotation,
                      'max_iters': max_iters, 'verbose': verbose,
                      'view': view}
        kwargs = dict(new_kwargs.items() + kwargs.items())

        self._fitting_function = self.fit_landmarked_image

        return self._fit_image_list(images, **kwargs)

    def _fit_image_list(self, images, **kwargs):
        n_images = len(images)

        fitting_list = []
        for j, i in enumerate(images):
            if kwargs['verbose']:
                print '- fitting image {} of {}'.format(j, n_images)
            fittings = self._fitting_function(i, **kwargs)
            fitting_list.append(fittings)

        return FittingList(fitting_list)


class AAMLKFitter(AAM, Fitter):

    def __init__(self, aam, lk_algorithm=AlternatingInverseCompositional,
                 residual=(LSIntensity,),
                 transform_cls=(OrthoMDTransform, SimilarityTransform),
                 n_shape=None, n_appearance=None):
        super(AAMLKFitter, self).__init__(
            aam.shape_model_list, aam.appearance_model_list,
            aam.reference_shape, aam.features, aam.downscale,
            aam.scaled_reference_frames, aam.transform_cls,
            aam.interpolator, aam.patch_size)

        self.residual = residual[0].type
        self.algorithm = lk_algorithm.type

        self._set_up(lk_algorithm=lk_algorithm, residual=residual,
                     transform_cls=transform_cls, n_shape=n_shape,
                     n_appearance=n_appearance)

    def _set_up(self, lk_algorithm=AlternatingInverseCompositional,
                residual=(LSIntensity,),
                transform_cls=(OrthoMDTransform, SimilarityTransform),
                n_shape=None, n_appearance=None):
        r"""
        Initializes the Lucas-Kanade fitting framework.

        Parameters
        -----------
        lk_algorithm: :class:`pybug.lucaskanade.appearance`, optional
            The Lucas-Kanade fitting algorithm to be used.

            Default: AlternatingInverseCompositional

        residual: :class:`pybug.lucaskanade.residual`, optional
            The type of residual to be used

            Default: 'LSIntensity'

        md_transform_cls: tuple, optional
                          (:class:`pybug.transform.ModelDrivenTransform`,
                           :class:`pybug.transform.Affine`)
            The first element of the tuple specifies the model driven
            transform to be used by the Lucas-Kanade objects. In case this
            is one of the global model driven transforms, the second element
            is the transform (from the affine family) to be used as a global
            transform.

            Default: (OrthoMDTransform, SimilarityTransform)

        n_shape: list, optional
            A list containing the number of shape components to be used at
            each level.

            Default: None

        n_appearance: list, optional
            A list containing the number of appearance components to be used
            at each level.

            Default: None
        """
        if n_shape is None:
            n_shape = [sm.n_active_components
                       for sm in self.shape_model_list]
        if n_appearance is None:
            n_appearance = [am.n_active_components
                            for am in self.appearance_model_list]

        if type(n_shape) is int:
            n_shape = [n_shape for _ in range(self.n_levels)]
        elif len(n_shape) is 1 and self.n_levels > 1:
            n_shape = [n_shape[1] for _ in range(self.n_levels)]
        elif len(n_shape) is not self.n_levels:
            raise ValueError('n_shape can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        if type(n_appearance) is int:
            n_appearance = [n_appearance for _ in range(self.n_levels)]
        elif len(n_appearance) is 1 and self.n_levels > 1:
            n_appearance = [n_appearance[1] for _ in range(self.n_levels)]
        elif len(n_appearance) is not self.n_levels:
            raise ValueError('n_appearance can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        self._lk_list = []
        for j, (am, sm) in enumerate(zip(self.appearance_model_list,
                                         self.shape_model_list)):

            if n_shape is not None:
                sm.n_active_components = n_shape[j]
            if n_appearance is not None:
                am.n_active_components = n_appearance[j]

            if transform_cls[0] is not ModelDrivenTransform:
                # ToDo: Do we need a blank (identity) method for Transforms?
                global_transform = transform_cls[1](np.eye(3, 3))
                md_transform = transform_cls[0](
                    sm, self.transform_cls, global_transform,
                    source=am.mean.landmarks['source'].lms)
            else:
                md_transform = transform_cls[0](
                    sm, self.transform_cls,
                    source=am.mean.landmarks['source'].lms)

            if len(residual) == 1:
                res = residual[0]()
            else:
                res = residual[0](residual[1])

            self._lk_list.append(lk_algorithm(am, res, md_transform))

    @property
    def _source(self):
        return self.shape_model_list[0].mean

    def _build_fitting(self, image, basic_fittings, affine_correction,
                       ground_truth=None, error_type='me_norm'):
        return AAMFitting(image, self, basic_fittings, affine_correction,
                          ground_truth=ground_truth, error_type=error_type)

    def _prepare_image(self, image, initial_target, gt_target=None):
        r"""
        Prepares an image to be be fitted by the AAM. The image is first
        rescaled wrt the reference landmarks, then smoothing or gaussian
        pyramid are computed and, finally, features extracted from each
        pyramidal element.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The original image to be fitted.

        initial_target: class:`pybug.shape.PointCloud`

        Returns
        -------
        image_list: :class:`pybug.image.masked.MaskedImage` list
            A list containing the images that will be passed to the fitting
            algorithms.
        """
        image.landmarks['initial_target'] = initial_target
        image = image.rescale_to_reference_landmarks(self.reference_shape,
                                                     group='initial_target')
        if gt_target:
            image.landmarks['gt_target'] = initial_target

        if self.scaled_reference_frames:
            pyramid = image.smoothing_pyramid(n_levels=self.n_levels,
                                              downscale=self.downscale)
        else:
            pyramid = image.gaussian_pyramid(n_levels=self.n_levels,
                                             downscale=self.downscale)

        image_list = [compute_features(i, self.features) for i in pyramid]
        image_list.reverse()

        return image_list

    def fit(self, image_list, initial_target, gt_target_list=None,
            max_iters=20, **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image_list: :class:`pybug.image.masked.MaskedImage` list
            A list containing the images to be fitted.

        initial_target: :class:`pybug.shape.PointCloud`
            The initial target from which the fitting will start.

        max_iters: int, optional
            The maximum number of iterations per level.

            Default: 20

        gt_target_list: :class:`pybug.shape.PointCloud` list,
                                  optional
            A list containing the ground truth targets for each of the
            images in image_list.

            Default: None

        Returns
        -------
        lk_fittings: :class:`pybug.lucasKanade.LKFitting` list
            A list containing the obtained
            :class:`pybug.lucasKanade.LKFitting` per level.
        """
        target = initial_target
        lk_fittings = []
        for j, (i, lk) in enumerate(zip(image_list, self._lk_list)):
            lk.transform.target = target

            lk_fitting = lk.align(i, lk.transform.as_vector(),
                                  max_iters=max_iters, **kwargs)

            if gt_target_list is not None:
                lk_fitting.gt_target = gt_target_list[j]
            lk_fittings.append(lk_fitting)

            target = lk_fitting.final_target
            if not self.scaled_reference_frames:
                Scale(self.downscale,
                      n_dims=lk.transform.n_dims).apply_inplace(target)

        return lk_fittings
