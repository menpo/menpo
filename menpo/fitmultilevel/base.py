from __future__ import division
import abc
import numpy as np
from copy import deepcopy

from menpo.fit.base import Fitter
from menpo.fit.fittingresult import FittingResultList
from menpo.transform import AlignmentAffine, Scale
from menpo.landmark import LandmarkGroup

from .fittingresult import MultilevelFittingResult
from .functions import noisy_align


class MultilevelFitter(Fitter):
    r"""
    Mixin that all MultilevelFitter objects must implement.
    """

    @abc.abstractproperty
    def reference_shape(self):
        r"""
        Returns the reference shape. Typically, the mean of shape model.
        """
        pass

    @abc.abstractproperty
    def feature_type(self):
        r"""
        Defines the feature computation function.
        """
        pass

    @abc.abstractproperty
    def n_levels(self):
        r"""
        Returns the number of levels used by fitter.
        """
        pass

    @abc.abstractproperty
    def downscale(self):
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

    @abc.abstractproperty
    def interpolator(self):
        r"""
        Returns the type of interpolator used by the fitter.
        """
        pass

    def fit_images(self, images, group=None, label='all',
                   initialization='from_gt_shape', noise_std=0.0,
                   rotation=False, max_iters=50, verbose=True, view=False,
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

        noise_std: float
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean
            Specifies whether in-plane rotation is to be used to produce the
            initial shape.

            Default: False

        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

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

        fittings = []
        for j, image in enumerate(images):
            if verbose:
                print '- fitting image {} of {}'.format(j, n_images)
            fittings.append(
                self.fit(image, group=group, label=label,
                         initialization=initialization, noise_std=noise_std,
                         rotation=rotation, max_iters=max_iters,
                         verbose=verbose, view=view, error_type=error_type,
                         **kwargs))

        return FittingResultList(fittings)

    def fit(self, image, group=None, label='all',
            initialization='from_gt_shape', noise_std=0.0, rotation=False,
            max_iters=50, verbose=True, view=False, error_type='me_norm',
            **kwargs):
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

        noise_std: float
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.0

        rotation: boolean
            Specifies whether in-plane rotation is to be used to produce the
            initial shape.

            Default: False

        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

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

        affine_correction = AlignmentAffine(initial_shapes[-1], initial_shape)

        fittings = self._fit(images, initial_shapes[0], max_iters=max_iters,
                             gt_shapes=gt_shapes, **kwargs)

        multiple_fitting = self._create_fitting(image, fittings,
                                                affine_correction,
                                                gt_shape=gt_shape,
                                                error_type=error_type)

        if verbose:
            multiple_fitting.print_fitting_info()
        if view:
            multiple_fitting.view_final_fitting(new_figure=True)

        return multiple_fitting

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
        reference_shape = self.reference_shape
        return noisy_align(reference_shape, gt_shape, noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

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

    def _create_fitting(self, image, fittings, affine_correction,
                        gt_shape=None, error_type='me_norm'):
        r"""
        Creates the :class: `pybug.aam.fitting.MultipleFitting` object
        associated with a particular Fitter objects.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The original image to be fitted.

        fittings: :class:`pybug.aam.fitting.BasicFitting` list
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
        return MultilevelFittingResult(image, self, fittings,
                                       affine_correction, gt_shape=gt_shape,
                                       error_type=error_type)

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
             **kwargs):
        r"""
        Fits the AAM to an image using Lucas-Kanade.

        Parameters
        -----------
        images: :class:`pybug.image.masked.MaskedImage` list
            The images to be fitted.

        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape from which the fitting will start.

        gt_shapes: :class:`pybug.shape.PointCloud` list, optional
            The original ground truth shapes associated to the images.

            Default: None

        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

        Returns
        -------
        fittings: :class:`pybug.aam.fitting` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        shape = initial_shape
        n_levels = self.n_levels

        if type(max_iters) is int:
            max_iters = [np.round(max_iters/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) is 1 and n_levels > 1:
            max_iters = [np.round(max_iters[0]/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) is not n_levels:
            raise ValueError('n_shape can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        gt = None
        fittings = []
        for j, (i, f, it) in enumerate(zip(images, self._fitters, max_iters)):
            if gt_shapes is not None:
                gt = gt_shapes[j]

            parameters = f.get_parameters(shape)
            fitting = f.fit(i, parameters, gt_shape=gt,
                            max_iters=it, **kwargs)
            fittings.append(fitting)

            shape = fitting.final_shape
            if self.scaled_levels:
                Scale(self.downscale,
                      n_dims=shape.n_dims).apply_inplace(shape)

        return fittings
