from __future__ import division
import abc
import numpy as np
from copy import deepcopy

from menpo.fit.base import Fitter
from menpo.transform import AlignmentAffine, Scale
from menpo.fitmultilevel.featurefunctions import compute_features

from .fittingresult import MultilevelFittingResult
from .functions import noisy_align, align_shape_with_bb


class MultilevelFitter(Fitter):
    r"""
    Mixin that all MultilevelFitter objects must implement.
    """

    @abc.abstractproperty
    def reference_shape(self):
        r"""
        Returns the reference shape. Typically, the mean of the shape model.
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
        Returns the number of levels used by the fitter.
        """
        pass

    @abc.abstractproperty
    def downscale(self):
        r"""
        Returns the downscale factor used by the fitter.
        """
        pass

    @abc.abstractproperty
    def pyramid_on_features(self):
        r"""
        Returns True if the pyramid is computed on the feature image and False
        if it is computed on the original (intensities) image and features are
        extracted at each level.
        """
        pass

    @abc.abstractproperty
    def interpolator(self):
        r"""
        Returns the type of interpolator used by the fitter.
        """
        pass

    def fit(self, image, initial_shape, max_iters=50, gt_shape=None,
            error_type='me_norm', verbose=False, view=False, **kwargs):
        r"""
        Fits a single image.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The image to be fitted.
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50
        gt_shape: PointCloud
            The groundtruth shape of the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'
        verbose: boolean, optional
            If True, it prints information related to the fitting results (such
            as: final error, convergence, ...).

            Default: False
        view: boolean, optional
            If True, the final fitting result will be visualized.

            Default: False

        **kwargs:

        Returns
        -------
        fitting_list: :map:`FittingResultList`
            A fitting result object.
        """
        # copy image
        image = deepcopy(image)

        # generate image pyramid
        images = self._prepare_image(image, initial_shape, gt_shape=gt_shape)

        # get ground truth shape per level
        if gt_shape:
            gt_shapes = [i.landmarks['gt_shape'].lms for i in images]
        else:
            gt_shapes = None

        # get initial shape per level
        initial_shapes = [i.landmarks['initial_shape'].lms for i in images]

        affine_correction = AlignmentAffine(initial_shapes[-1], initial_shape)

        # execute multilevel fitting
        fitting_results = self._fit(images, initial_shapes[0],
                                    max_iters=max_iters,
                                    gt_shapes=gt_shapes, **kwargs)

        # store result
        multilevel_fitting_result = self._create_fitting_result(
            image, fitting_results, affine_correction, gt_shape=gt_shape,
            error_type=error_type)

        if verbose:
            print multilevel_fitting_result
        if view:
            multilevel_fitting_result.view_final_fitting(new_figure=True)

        return multilevel_fitting_result

    def perturb_shape(self, gt_shape, noise_std=0.04, rotation=False):
        r"""
        Generates an initial shape by adding gaussian noise to the perfect
        similarity alignment between the ground truth and reference_shape.

        Parameters
        -----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        noise_std: float, optional
            The standard deviation of the gaussian noise used to produce the
            initial shape.

            Default: 0.04
        rotation: boolean, optional
            Specifies whether ground truth in-plane rotation is to be used
            to produce the initial shape.

            Default: False

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """
        reference_shape = self.reference_shape
        return noisy_align(reference_shape, gt_shape, noise_std=noise_std,
                           rotation=rotation).apply(reference_shape)

    def obtain_shape_from_bb(self, bounding_box):
        r"""
        Generates an initial shape given a bounding box detection.

        Parameters
        -----------
        bounding_box: (2, 2) ndarray
            The bounding box specified as:

                np.array([[x_min, y_min], [x_max, y_max]])

        Returns
        -------
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape.
        """
        reference_shape = self.reference_shape
        return align_shape_with_bb(reference_shape,
                                   bounding_box).apply(reference_shape)

    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        The image is first rescaled wrt the reference_landmarks and then the
        gaussian pyramid is computed. Depending on the pyramid_on_features
        flag, the pyramid is either applied on the feature image or
        features are extracted at each pyramidal level.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The image to be fitted.
        initial_shape: class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The original ground truth shape associated to the image.

            Default: None

        Returns
        -------
        images: list of :class:`menpo.image.masked.MaskedImage`
            List of images, each being the result of applying the pyramid.
        """
        # rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image.landmarks['initial_shape'] = initial_shape
        image = image.rescale_to_reference_shape(
            self.reference_shape, group='initial_shape',
            interpolator=self.interpolator)

        # attach given ground truth shape
        if gt_shape:
            image.landmarks['gt_shape'] = gt_shape

        # apply pyramid
        if self.n_levels > 1:
            if self.pyramid_on_features:
                # compute features at highest level
                feature_image = compute_features(image, self.feature_type[0])

                # apply pyramid on feature image
                pyramid = feature_image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # get rescaled feature images
                images = list(pyramid)
            else:
                # create pyramid on intensities image
                pyramid = image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # compute features at each level
                images = [compute_features(
                    i, self.feature_type[self.n_levels - j - 1])
                    for j, i in enumerate(pyramid)]
            images.reverse()
        else:
            images = [compute_features(image, self.feature_type[0])]
        return images

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None, error_type='me_norm'):
        r"""
        Creates the :class: `menpo.aam.fitting.MultipleFitting` object
        associated with a particular Fitter object.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The original image to be fitted.
        fitting_results: :class:`menpo.fit.fittingresult.FittingResultList`
            A list of basic fitting objects containing the state of the
            different fitting levels.
        affine_correction: :class: `menpo.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        fitting: :class:`menpo.fitmultilevel.fittingresult.MultilevelFittingResult`
            The fitting object that will hold the state of the fitter.
        """
        return MultilevelFittingResult(image, self, fitting_results,
                                       affine_correction, gt_shape=gt_shape,
                                       error_type=error_type)

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
             **kwargs):
        r"""
        Fits the fitter to the multilevel pyramidal images.

        Parameters
        -----------
        images: :class:`menpo.image.masked.MaskedImage` list
            The images to be fitted.
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shapes: :class:`menpo.shape.PointCloud` list, optional
            The original ground truth shapes associated to the multilevel
            images.

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
        fitting_results: :class:`menpo.fit.fittingresult.FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        shape = initial_shape
        n_levels = self.n_levels

        # check max_iters parameter
        if type(max_iters) is int:
            max_iters = [np.round(max_iters/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) == 1 and n_levels > 1:
            max_iters = [np.round(max_iters[0]/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) != n_levels:
            raise ValueError('max_iters can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        # fitting
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
            Scale(self.downscale, n_dims=shape.n_dims).apply_inplace(shape)

        return fitting_results
