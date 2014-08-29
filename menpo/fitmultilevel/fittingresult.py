from __future__ import division

from menpo.transform import Scale
from menpo.fit.fittingresult import FittingResult


class MultilevelFittingResult(FittingResult):
    r"""
    Class that holds the state of a :map:`MultilevelFitter` object before,
    during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.

    multilevel_fitter : :map:`MultilevelFitter`
        The multilevel fitter object used to fit the image.

    fitting_results : `list` of :map:`FittingResult`
        The list of fitting results.

    affine_correction : :map:`Affine`
        The affine transform between the initial shape of the highest
        pyramidal level and the initial shape of the original image

    gt_shape : class:`PointCloud`, optional
        The ground truth shape associated to the image.

    error_type : 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between is computed.
    """
    def __init__(self, image, multiple_fitter, fitting_results,
                 affine_correction, gt_shape=None):
        super(MultilevelFittingResult, self).__init__(
            image, multiple_fitter, gt_shape=gt_shape)
        self.fitting_results = fitting_results
        self._affine_correction = affine_correction

    @property
    def n_levels(self):
        r"""
        The number of levels of the fitter object.

        :type: `int`
        """
        return self.fitter.n_levels

    @property
    def downscale(self):
        r"""
        The downscale factor used by the multiple fitter.

        :type: `float`
        """
        return self.fitter.downscale

    @property
    def n_iters(self):
        r"""
        The total number of iterations used to fitter the image.

        :type: `int`
        """
        n_iters = 0
        for f in self.fitting_results:
            n_iters += f.n_iters
        return n_iters

    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : `boolean`, optional
            Whether the result is returned as a `list` of :map:`PointCloud` or
            a `list` of `ndarrays`.

        Returns
        -------
        shapes : `list` of :map:`PointCoulds` or `list` of `ndarray`
            A list containing the fitted shapes at each iteration of
            the fitting procedure.
        """
        n = self.n_levels - 1
        shapes = []
        for j, f in enumerate(self.fitting_results):
            transform = Scale(self.downscale**(n-j), f.final_shape.n_dims)
            for t in f.shapes(as_points=as_points):
                t = transform.apply(t)
                shapes.append(self._affine_correction.apply(t))

        return shapes

    @property
    def final_shape(self):
        r"""
        The final fitted shape.

        :type: :map:`PointCloud`
        """
        return self._affine_correction.apply(
            self.fitting_results[-1].final_shape)

    @property
    def initial_shape(self):
        r"""
        The initial shape from which the fitting started.

        :type: :map:`PointCloud`
        """
        n = self.n_levels - 1
        initial_shape = self.fitting_results[0].initial_shape
        Scale(self.downscale ** n, initial_shape.n_dims).apply_inplace(
            initial_shape)

        return self._affine_correction.apply(initial_shape)

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.

        type: :map:`PointCloud`
        """
        self._gt_shape = value

    # TODO: this should print more information than just this...
    # image, fitter, n_iters run, ...
    def __str__(self):
        # string about features
        if self.fitter.pyramid_on_features:
            if isinstance(self.fitter.feature_type[0], str):
                feat_str = self.fitter.feature_type[0]
            elif self.fitter.feature_type[0] is None:
                feat_str = "no"
            else:
                feat_str = self.fitter.feature_type[0].__name__
        else:
            feat_str = []
            for j in range(self.n_levels):
                if isinstance(self.fitter.feature_type[j], str):
                    feat_str.append(self.fitter.feature_type[j])
                elif self.fitter.feature_type[j] is None:
                    feat_str.append("none")
                else:
                    feat_str.append(self.fitter.feature_type[j].__name__)
        out = "{0} Fitting Result\n" \
              " - Initial error: {1:.4f}\n" \
              " - Final error: {2:.4f}\n" \
              " - {3} optimization with {4} pyramid levels, {5} iterations " \
              "and using {6} features.".format(
              self.fitter.aam._str_title, self.initial_error(),
              self.final_error(), self.fitter._fitters[0].algorithm,
              self.n_levels, self.n_iters, feat_str)
        return out


class AAMMultilevelFittingResult(MultilevelFittingResult):
    r"""
    Class that holds the state of a :map:`AAMFitter` object before,
    during and after it has fitted a particular image.

    Parameters
    -----------
    image : :map:`Image` or subclass
        The fitted image.

    multilevel_fitter : :map:`MultilevelFitter`
        The multilevel fitter object used to fit the image.

    fitting_results : `list` of :map:`FittingResult`
        The list of fitting results.

    affine_correction : :map:`Affine`
        The affine transform between the initial shape of the highest
        pyramidal level and the initial shape of the original image

    gt_shape : class:`PointCloud`, optional
        The ground truth shape associated to the image.

    error_type : 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between is computed.
    """
    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.

        :type: `list` of `float`
        """
        raise ValueError('costs not implemented yet.')
        #return self._flatten_out([f.costs for f in self.basic_fittings])

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.

        :type: `float`
        """
        return self.fitting_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.

        :type: `float`
        """
        return self.fitting_results[0].initial_cost

    @property
    def warped_images(self):
        r"""
        The list containing the warped images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        mask = self.fitting_results[-1].fitter.template.mask
        transform = self.fitting_results[-1].fitter.transform
        interpolator = self.fitting_results[-1].fitter.interpolator

        warped_images = []
        for s in self.shapes():
            transform.set_target(s)
            image = self.image.warp_to(mask, transform,
                                       interpolator=interpolator)
            warped_images.append(image)

        return warped_images

    @property
    def appearance_reconstructions(self):
        r"""
        The list containing the appearance reconstruction obtained at
        each fitting iteration.

        :type: `list` of :map:`Image` or subclass
        """
        return _flatten_out(
            [f.appearance_reconstructions for f in self.fitting_results])

    @property
    def error_images(self):
        r"""
        The list containing the error images obtained at each fitting
        iteration.

        :type: `list` of :map:`Image` or subclass
        """
        return _flatten_out(
            [f.error_images for f in self.fitting_results])


def _flatten_out(list_of_lists):
    return [i for l in list_of_lists for i in l]
