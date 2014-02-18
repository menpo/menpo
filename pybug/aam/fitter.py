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
from pybug.aam.functions import noisy_align, compute_features
from pybug.aam.fitting import FittingList, AAMFitting


class Fitter(object):
    r"""
    Interface that all fitter objects must implement.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def _default_shape(self):
        r"""
        Returns the default shape. Typically, the mean of shape model.
        """
        pass


    @abc.abstractproperty
    def _downscale(self):
        r"""
        Returns the downscale factor used by the fitter.
        """
        pass

    @abc.abstractproperty
    def scaled_levels(self):
        r"""
        Returns True if the shape results returned by the basic fittings
        must be scaled.
        """
        pass

    @abc.abstractmethod
    def _set_up(self, **kwargs):
        r"""
        Sets up the Fitter object. Highly dependent on the type of fitter
        object.
        """
        pass

    @abc.abstractmethod
    def _fitting(self, image, basic_fittings, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        r"""
        Creates the :class: `pybug.aam.fitting.Fitting` object associated
        with a particular Fitter class.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The original image to be fitted.

        basic_fittings: :class:`pybug.aam.fitting.BasicFitting` list
            A list of basic fitting objects containing the state of the
            different fitting levels.

        affine_correction: :class: `pybug.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.

        gt_shape: class:`pybug.shape.PointCloud`, optional
            The ground truth shape associated to the image.

            Default: None

        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        fitting: :class:`pybug.aam.Fitting`
            The fitting object that will hold the state of the fitter.
        """
        pass

    @abc.abstractmethod
    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        Prepares an image to be fitted.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The original image to be fitted.

        initial_shape: class:`pybug.shape.PointCloud`
            The initial shape from which the fitting will start.

        gt_shape: class:`pybug.shape.PointCloud`, optional
            The original ground truth shape associated to the image.

            Default: None

        Returns
        -------
        images: :class:`pybug.image.masked.MaskedImage` list
            A list containing the images that will be used by the fitting
            algorithms.
        """
        pass

    @abc.abstractmethod
    def _fit(self, image, initial_shape, gt_shape=None,
             max_iters=20, **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage` list
            The images to be fitted.

        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape from which the fitting will start.

        gt_shape: :class:`pybug.shape.PointCloud` list, optional
            The original ground truth shape associated to the image.

            Default: None

        max_iters: int, optional
            The maximum number of iteration per fitting level.

            Default: 20

        Returns
        -------
        fittings: :class:`pybug.aam.fitting` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        pass

    def _detect_shape(self, noise_std=0.0, rotation=False):
        r"""
        Generates an initial shape by automatically detecting the object
        being modelled (typically faces) in the image. This method should be
        wired to future face and object detection algorithms.

        Parameters
        -----------
        noise_std: float, optional
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean, optional
            Specifies whether rotation is to be used to produce the initial
            shape.

            Default: False

        Returns
        -------
        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape.
        """
        raise ValueError('_detect_shape not implemented yet')

    def _noisy_align_from_gt_shape(self, gt_shape, noise_std=0.0,
                                   rotation=False):
        r"""
        Generates an initial shape by adding gaussian noise  to
        the perfect similarity alignment between the ground truth
        and default shape.

        Parameters
        -----------
        gt_shape: :class:`pybug.shape.PointCloud` list
            The ground truth shape.

        noise_std: float, optional
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean, optional
            Specifies whether ground truth in-plane rotation is to be used
            to produce the initial shape.

            Default: False

        Returns
        -------
        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape.
        """
        default_shape = self._default_shape
        return noisy_align(default_shape, gt_shape, noise_std=noise_std,
                           rotation=rotation).apply(default_shape)

    def fit_image(self, image, group=None, label='all',
                  initialization='from_gt_shape', runs=1, noise_std=0.0,
                  rotation=False, max_iters=20, verbose=True, view=False,
                  error_type='me_norm', **kwargs):
        r"""
        Fits a single image.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The image to be fitted.

        group: string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        initialization: 'from_gt_shape' or 'detection', optional
            The type of initialization to be used for fitting the image.

            Default: 'from_gt_shape'

        runs: int, optional
            The number of times the image must be fitted.

            Default: 1

        noise_std: float
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean
            Specifies whether in-plane rotation is to be used to produce the
            initial shape.

            Default: False

        max_iters: int, optional
            The maximum number of iteration per fitting level.

            Default: 20

        verbose: boolean
            Whether or not to print information related to the fitting
            results (such as: final error, convergence, ...).

            Default: True

        view: boolean
            Whether or not the fitting results are to be displayed.

            Default: False

        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        FittingList: :class:`pybug.aam.fitting.FittingList`
            A fitting list object containing the fitting objects associated
            to each run.
        """
        image = deepcopy(image)

        if isinstance(image.landmarks[group][label], LandmarkGroup):
            gt_shape = image.landmarks[group][label].lms
        else:
            if group or label is not 'all':
                raise ValueError('The specified group {} and/or '
                                 'label {} do not exist'.format(group,
                                                                 label))
            elif initialization is not 'detection':
                raise ValueError('Initialization method {} cannot '
                                 'be used because the image is not '
                                 'landmarked'.format(initialization))
            gt_shape = None


        fittings = []
        for _ in range(runs):

            if initialization is 'from_gt_shape':
                initial_shape = self._noisy_align_from_gt_shape(
                    gt_shape, noise_std=noise_std, rotation=rotation)
            elif type is 'detection':
                initial_shape = self._detect_shape(
                    noise_std=noise_std, rotation=rotation)
            else:
                raise ValueError('Unknown initialization string selected. '
                                 'Valid options are: "from_gt_shape", '
                                 '"detection"')

            images = self._prepare_image(image, initial_shape,
                                         gt_shape=gt_shape)

            if gt_shape:
                gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
            else:
                gt_shapes = None

            initial_shapes = [i.landmarks['initial_shape'].lms
                              for i in images]

            affine_correction = AffineTransform.align(initial_shapes[-1],
                                                      initial_shape)

            basic_fittings = self._fit(images, initial_shapes[0],
                                       max_iters=max_iters,
                                       gt_shapes=gt_shapes,
                                       **kwargs)

            fitting = self._fitting(image, basic_fittings, affine_correction,
                                    gt_shape=gt_shape, error_type=error_type)
            fittings.append(fitting)

            if verbose:
                fitting.print_fitting_info()
            if view:
                fitting.view_final_fitting(new_figure=True)

        return FittingList(fittings)

    def fit_images(self, images, group=None, label='all',
                   initialization='from_gt_shape', runs=5, noise_std=0.0,
                   rotation=False, max_iters=20, verbose=True, view=False,
                   error_type='me_norm', **kwargs):
        r"""
        Fits a list of images.

        Parameters
        -----------
        images: list of :class:`pybug.image.masked.MaskedImage`
            The list of images to be fitted.

        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None

        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'

        initialization: 'from_gt_shape' or 'detection', optional
            The type of initialization to be used for fitting the image.

            Default: 'from_gt_shape'

        runs: int, optional
            The number of times the image must be fitted.

            Default: 1

        noise_std: float
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean
            Specifies whether in-plane rotation is to be used to produce the
            initial shape.

            Default: False

        max_iters: int, optional
            The maximum number of iteration per fitting level.

            Default: 20

        verbose: boolean
            Whether or not to print information related to the fitting
            results (such as: final error, convergence, ...).

            Default: True

        view: boolean
            Whether or not the fitting results are to be displayed.

            Default: False

        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        FittingList: :class:`pybug.aam.fitting.FittingList`
            A fitting list object containing a fitting list object
            associated to each image.
        """
        n_images = len(images)

        fitting_list = []
        for j, image in enumerate(images):
            if verbose:
                print '- fitting image {} of {}'.format(j, n_images)
            fittings = self.fit_image(image, group=group, label=label,
                                      initialization=initialization,
                                      runs=runs, noise_std=noise_std,
                                      rotation=rotation, max_iters=max_iters,
                                      verbose=verbose, view=view,
                                      error_type=error_type, **kwargs)
            fitting_list.append(fittings)

        return FittingList(fitting_list)


class AAMFitter(Fitter):
    r"""
    An abstract class Active Appearance Models Fitters.

    Parameters
    -----------
    aam: :class:`pybug.aam.AAM`
        The Active Appearance Model to be use.

    """

    def __init__(self, aam):
        self.aam = aam

    @property
    def _default_shape(self):
        return self.aam.shape_models[0].mean

    @property
    def _downscale(self):
        return self.aam.downscale

    @property
    def scaled_levels(self):
        return self.aam.scaled_reference_frames


class LucasKanadeAAMFitter(AAMFitter):
    r"""
    Lucas-Kanade based Fitter for Active Appearance Models.

    Parameters
    -----------
    aam: :class:`pybug.aam.AAM`
        The Active Appearance Model to be use.

    lk_object_cls: :class:`pybug.lucaskanade.appearance`, optional
            The Lucas-Kanade class to be used.

            Default: AlternatingInverseCompositional

    residual_cls: :class:`pybug.lucaskanade.residual`, optional
        The residual class to be used

        Default: 'LSIntensity'

    md_transform_cls: :class:`pybug.transform.ModelDrivenTransform`,
                      optional
        The model driven transform class to be used.

        Default: OrthoMDTransform

    global_transform_cls: :class:`pybug.transform.affine`, optional
        The global transform class to be used by the previous
        md_transform_cls. Currently, only
        :class:`pybug.transform.affine.Similarity` is supported.

        Default: SimilarityTransform

    n_shape: list, optional
        The number of shape components to be used per fitting level.
        If None, for each shape model n_active_components will be used.

        Default: None

    n_appearance: list, optional
        The number of appearance components to be used per fitting level.
        If None, for each appearance model n_active_components will be used.

        Default: None
    """

    def __init__(self, aam, lk_algorithm=AlternatingInverseCompositional,
                 residual=LSIntensity, md_transform_cls=OrthoMDTransform,
                 global_transform_cls=SimilarityTransform, n_shape=None,
                 n_appearance=None):
        super(LucasKanadeAAMFitter, self).__init__(aam)
        self.algorithm = lk_algorithm.type
        self.residual = residual.type

        self._set_up(lk_object_cls=lk_algorithm, residual_cls=residual,
                     md_transform_cls=md_transform_cls,
                     global_transform_cls=global_transform_cls,
                     n_shape=n_shape, n_appearance=n_appearance)

    def _set_up(self, lk_object_cls=AlternatingInverseCompositional,
                residual_cls=LSIntensity, md_transform_cls=OrthoMDTransform,
                global_transform_cls=SimilarityTransform, n_shape=None,
                n_appearance=None):
        r"""
        Re-initializes the Lucas-Kanade based fitting.

        Parameters
        -----------
        lk_object_cls: :class:`pybug.lucaskanade.appearance`, optional
            The Lucas-Kanade class to be used.

            Default: AlternatingInverseCompositional

        residual_cls: :class:`pybug.lucaskanade.residual`, optional
            The residual class to be used

            Default: 'LSIntensity'

        md_transform_cls: :class:`pybug.transform.ModelDrivenTransform`,
                          optional
            The model driven transform class to be used.

            Default: OrthoMDTransform

        global_transform_cls: :class:`pybug.transform.affine`, optional
            The global transform class to be used by the previous
            md_transform_cls. Currently, only
            :class:`pybug.transform.affine.Similarity` is supported.

            Default: SimilarityTransform

        n_shape: list, optional
            The number of shape components to be used per fitting level.
            If None, for each shape model n_active_components will be used.

            Default: None

        n_appearance: list, optional
            The number of appearance components to be used per fitting level.
            If None, for each appearance model n_active_components will be used.

            Default: None
        """
        if n_shape is None:
            n_shape = [sm.n_active_components
                       for sm in self.aam.shape_models]
        if n_appearance is None:
            n_appearance = [am.n_active_components
                            for am in self.aam.appearance_models]

        if type(n_shape) is int:
            n_shape = [n_shape for _ in range(self.aam.n_levels)]
        elif len(n_shape) is 1 and self.aam.n_levels > 1:
            n_shape = [n_shape[1] for _ in range(self.aam.n_levels)]
        elif len(n_shape) is not self.aam.n_levels:
            raise ValueError('n_shape can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.aam.n_levels))

        if type(n_appearance) is int:
            n_appearance = [n_appearance for _ in range(self.aam.n_levels)]
        elif len(n_appearance) is 1 and self.aam.n_levels > 1:
            n_appearance = [n_appearance[1] for _ in range(self.aam.n_levels)]
        elif len(n_appearance) is not self.aam.n_levels:
            raise ValueError('n_appearance can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.aam.n_levels))

        self._lk_objects = []
        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            if n_shape is not None:
                sm.n_active_components = n_shape[j]
            if n_appearance is not None:
                am.n_active_components = n_appearance[j]

            if md_transform_cls is not ModelDrivenTransform:
                # ToDo: Do we need a blank (identity) method for Transforms?
                global_transform = global_transform_cls(np.eye(3, 3))
                md_transform = md_transform_cls(
                    sm, self.aam.transform_cls, global_transform,
                    source=am.mean.landmarks['source'].lms)
            else:
                md_transform = md_transform_cls(
                    sm, self.aam.transform_cls,
                    source=am.mean.landmarks['source'].lms)

            self._lk_objects.append(lk_object_cls(am, residual_cls(),
                                                  md_transform))

    def _fitting(self, image, basic_fittings, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        return AAMFitting(image, self, basic_fittings, affine_correction,
                          gt_shape=gt_shape, error_type=error_type)

    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        The image is first rescaled wrt the reference landmarks,
        then smoothing or gaussian pyramid are computed and, finally,
        features extracted from each pyramidal element.
        """
        image.landmarks['initial_shape'] = initial_shape
        image = image.rescale_to_reference_shape(self.aam.reference_shape,
                                                     group='initial_shape')
        if gt_shape:
            image.landmarks['gt_shape'] = initial_shape

        if self.aam.scaled_reference_frames:
            pyramid = image.smoothing_pyramid(n_levels=self.aam.n_levels,
                                              downscale=self.aam.downscale)
        else:
            pyramid = image.gaussian_pyramid(n_levels=self.aam.n_levels,
                                             downscale=self.aam.downscale)

        images = [compute_features(i, self.aam.features) for i in pyramid]
        images.reverse()

        return images

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=20,
             **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        images: :class:`pybug.image.masked.MaskedImage` list
            A list containing the images to be fitted.

        initial_shapes: :class:`pybug.shape.PointCloud`
            The initial shape from which the fitting will start.

        gt_shapes: :class:`pybug.shape.PointCloud` list, optional
            A list containing the original ground truth shape associated to
            each fitting level.

            Default: None

        max_iters: int, optional
            The maximum number of iterations per fitting level.

            Default: 20

        Returns
        -------
        lk_fittings: :class:`pybug.lucasKanade.LKFitting` list
            A list containing the fitting objects that hold the state of each
            fitting level.
        """
        shape = initial_shape
        lk_fittings = []
        for j, (i, lk) in enumerate(zip(images, self._lk_objects)):
            lk.transform.target = shape

            lk_fitting = lk.align(i, lk.transform.as_vector(),
                                  max_iters=max_iters, **kwargs)

            if gt_shapes is not None:
                lk_fitting.gt_shape = gt_shapes[j]
            lk_fittings.append(lk_fitting)

            shape = lk_fitting.final_shape
            if not self.aam.scaled_reference_frames:
                Scale(self.aam.downscale,
                      n_dims=lk.transform.n_dims).apply_inplace(shape)

        return lk_fittings


class RegressionAAMFitter(AAMFitter):
    r"""
    Regression based Fitter for Active Appearance Models.

    Parameters
    -----------
    aam: :class:`pybug.aam.AAM`
        The Active Appearance Model to be use.
    """

    def __init__(self, aam):
        super(RegressionAAMFitter, self).__init__(aam)
        raise ValueError('RegressionAAMFitter not implemented yet.')
