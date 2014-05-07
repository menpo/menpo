from __future__ import division
import abc
import numpy as np
from copy import deepcopy

from menpo.fit.base import Fitter
from menpo.transform import AlignmentAffine, Scale

from .fittingresult import MultilevelFittingResult
from .functions import noisy_align, align_shape_with_bb


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
        Returns True if the shape results returned by the basic fitting_results
        must be scaled.
        """
        pass

    @abc.abstractproperty
    def interpolator(self):
        r"""
        Returns the type of interpolator used by the fitter.
        """
        pass

    def fit(self, image, initial_shape, max_iters=50, gt_shape=None,
            error_type='me_norm', verbose=True, view=False, **kwargs):
        r"""
        Fits a single image.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The image to be fitted.

        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.

        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        verbose: boolean, optional
            Whether or not to print information related to the fitting
            results (such as: final error, convergence, ...).

            Default: True

        view: boolean, optional
            Whether or not the fitting results are to be displayed.

            Default: False

        **kwargs:

        Returns
        -------
        FittingList: :class:`pybug.aam.fitting.FittingList`
            A fitting list object containing the fitting objects associated
            to each run.
        """
        image = deepcopy(image)
        images = self._prepare_image(image, initial_shape, gt_shape=gt_shape)

        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        initial_shapes = [i.landmarks['initial_shape'].lms
                          for i in images]

        affine_correction = AlignmentAffine(initial_shapes[-1], initial_shape)

        fitting_results = self._fit(images, initial_shapes[0],
                                    max_iters=max_iters,
                                    gt_shapes=gt_shapes, **kwargs)

        multilevel_fitting_result = self._create_fitting_result(
            image, fitting_results, affine_correction, gt_shape=gt_shape,
            error_type=error_type)

        if verbose:
            multilevel_fitting_result.print_fitting_info()
        if view:
            multilevel_fitting_result.view_final_fitting(new_figure=True)

        return multilevel_fitting_result

    def perturb_shape(self, gt_shape, noise_std=0.04, rotation=False):
        r"""
        Generates an initial shape by adding gaussian noise  to
        the perfect similarity alignment between the ground truth
        and reference_shape.

        Parameters
        -----------
        gt_shape: :class:`pybug.shape.PointCloud` list
            The ground truth shape.

        noise_std: float, optional
            The std of the gaussian noise used to produce the initial shape.

            Default: 0.04

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

    def obtain_shape_from_bb(self, bounding_box):
        r"""
        Generates an initial shape giving a bounding box detection.

        Parameters
        -----------
        bounding_box: (4,) ndarray
            The bounding box.

        Returns
        -------
        initial_shape: :class:`pybug.shape.PointCloud`
            The initial shape.
        """
        reference_shape = self.reference_shape
        return align_shape_with_bb(reference_shape,
                                   bounding_box).apply(reference_shape)

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

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None, error_type='me_norm'):
        r"""
        Creates the :class: `pybug.aam.fitting.MultipleFitting` object
        associated with a particular Fitter objects.

        Parameters
        -----------
        image: :class:`pybug.image.masked.MaskedImage`
            The original image to be fitted.

        fitting_results: :class:`pybug.aam.fitting.BasicFitting` list
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
        return MultilevelFittingResult(image, self, fitting_results,
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
        fitting_results: :class:`pybug.aam.fitting` list
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
        fitting_results = []
        for j, (i, f, it) in enumerate(zip(images, self._fitters, max_iters)):
            if gt_shapes is not None:
                gt = gt_shapes[j]

            parameters = f.get_parameters(shape)
            fitting_result = f.fit(i, parameters, gt_shape=gt, max_iters=it,
                                   **kwargs)
            fitting_results.append(fitting_result)

            shape = fitting_result.final_shape
            if self.scaled_levels:
                Scale(self.downscale,
                      n_dims=shape.n_dims).apply_inplace(shape)

        return fitting_results
