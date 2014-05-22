from __future__ import division
import numpy as np

from menpo.transform import Scale
from menpo.visualize.base import GraphPlotter, MultipleImageViewer
from menpo.fit.fittingresult import FittingResult


class MultilevelFittingResult(FittingResult):
    r"""
    Object that holds the state of a MultipleFitter object (to which it is
    linked) after it has fitted a particular image.

    Parameters
    -----------
    image: :class:`menpo.image.masked.MaskedImage`
        The fitted image.

    multiple_fitter: :class:`menpo.fitter.base.Fitter`
        The fitter object used to fitter the image.

    fitting_results: :class:`menpo.fitter.fittingresult.FittingResult` list
        A list of basic fitting objects.

    affine_correction: :class: `menpo.transforms.affine.Affine`
        An affine transform that maps the result of the top resolution
        fitting level to the space scale of the original image.

    gt_shape: class:`menpo.shape.PointCloud`, optional
        The ground truth shape associated to the image.

        Default: None

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, image, multiple_fitter, fitting_results, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        self._error_stop = None  # Internal attribute of error_type setter
        self.fitting_results = fitting_results
        self._affine_correction = affine_correction
        super(MultilevelFittingResult, self).__init__(
            image, multiple_fitter, gt_shape=gt_shape, error_type=error_type)

    @property
    def n_levels(self):
        r"""
        Returns the number of levels of the fitter object.
        """
        return self.fitter.n_levels

    @property
    def downscale(self):
        r"""
        Returns the downscale factor used by the multiple fitter.
        """
        return self.fitter.downscale

    @property
    def scaled_levels(self):
        r"""
        Returns True if the shape results returned by the basic fitting_results
        must be scaled.
        """
        return True  # self.fitter.scaled_levels

    @property
    def fitted(self):
        r"""
        Returns the fitted state of each fitting object.
        """
        return [f.fitted for f in self.fitting_results]

    @FittingResult.error_type.setter
    def error_type(self, error_type):
        r"""
        Sets the error type according to a set of predefined options.
        """
        if error_type == 'me_norm':
            for f in self.fitting_results:
                f.error_type = error_type
            self._error_stop = 0.1
            self._error_text = 'Point-to-point error normalized by object ' \
                               'size'
        elif error_type == 'me':
            NotImplementedError("erro_type 'me' not implemented yet")
        elif error_type == 'rmse':
            NotImplementedError("error_type 'rmse' not implemented yet")
        else:
            raise ValueError("Unknown error_type string selected. Valid"
                             "options are: 'me_norm', 'me', 'rmse'")
        self._error_type = error_type

    @property
    def n_iters(self):
        r"""
        Returns the total number of iterations used to fitter the image.
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
        as_points: boolean, optional
            Whether the results is returned as a list of PointClouds or
            ndarrays.

            Default: False

        Returns
        -------
        shapes: :class:`menpo.shape.PointCoulds or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """
        n = self.n_levels - 1

        shapes = []
        for j, f in enumerate(self.fitting_results):
            if self.scaled_levels:
                transform = Scale(self.downscale**(n-j), 2)
                for t in f.shapes(as_points=as_points):
                    transform.apply_inplace(t)
                    shapes.append(self._affine_correction.apply(t))
            else:
                for t in f.shapes(as_points=as_points):
                    shapes.append(self._affine_correction.apply(t))

        return shapes

    @property
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """
        return self._affine_correction.apply(
            self.fitting_results[-1].final_shape)

    @property
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """
        n = self.n_levels - 1

        initial_shape = self.fitting_results[0].initial_shape
        if self.scaled_levels:
            Scale(self.downscale ** n,
                  initial_shape.n_dims).apply_inplace(initial_shape)

        return self._affine_correction.apply(initial_shape)

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        self._gt_shape = value

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the error evolution throughout the fitting.
        """
        if self.gt_shape is not None:
            title = 'Error evolution'
            legend = [self.algorithm]
            x_label = 'Number of iterations'
            y_label = self._error_text
            errors = self.errors
            x_limit = self.n_iters + self.n_levels
            axis_limits = [0, x_limit, 0, np.max(errors)]
            return GraphPlotter(figure_id, new_figure, range(0, x_limit),
                                [errors], title=title, legend=legend,
                                x_label=x_label, y_label=y_label,
                                axis_limits=axis_limits).render(**kwargs)
        else:
            raise ValueError('Ground truth shape has not been set, error '
                             'cannot be plotted')

    def __str__(self):
        out = "Initial error: {0:.4f}\nFinal error: {1:.4f}".format(
            self.initial_error, self.final_error)
        return out


class AAMMultilevelFittingResult(MultilevelFittingResult):
    r"""
    Object let us recover the state of an AAM Fitter after the latter has
    fitted a particular image.

    Parameters
    -----------
    image: :class:`pybug.image.masked.MaskedImage`
        The fitted image.

    aam_fitter: :class:`pybug.aam.fitter.AAMFitter`
        The aam_fitter object used to fitter the image.

    basic_fittings: :class:`pybug.aam.fitting.BasicFitting` list
        A list of basic fitting objects.

    _affine_correction: :class: `pybug.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.

    gt_shape: class:`pybug.shape.PointCloud`, optional
        The ground truth shape associated to the image.

        Default: None

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    @property
    def residual(self):
        r"""
        Returns the type of residual used by the basic fitter associated to
        each basic fitting.
        """
        # TODO: ensure that all basic_fitting residuals are the same?
        return self.fitting_results[-1].residual.type

    @property
    def costs(self):
        r"""
        Returns a list containing the cost at each fitting iteration.
        """
        raise ValueError('costs not implemented yet.')
        #return self._flatten_out([f.costs for f in self.basic_fittings])

    @staticmethod
    def _flatten_out(list_of_lists):
        return [i for l in list_of_lists for i in l]

    @property
    def final_cost(self):
        r"""
        Returns the final fitting cost.
        """
        return self.fitting_results[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.
        """
        return self.fitting_results[0].initial_cost

    def warped_images(self, from_basic_fittings=False, as_pixels=False):
        r"""
        Generates a list containing the warped images obtained at each fitting
        iteration.

        Parameters
        -----------
        from_basic_fittings : `boolean`, optional
            If ``True``, the returned transform per iteration is used to warp
            the internal image representation used by each basic fitter.
            If ``False``, the transforms are used to warp original image.

        as_pixels : `boolean`, optional
            Whether the result is returned as a list of :map:`Image` or
            `ndarray`.

        Returns
        -------
        warped_images : :map:`MaskedImage` or `ndarray` list
            A list containing the warped images obtained at each fitting
            iteration.
        """
        if from_basic_fittings:
            return self._flatten_out([f.warped_images(as_pixels=as_pixels)
                                      for f in self.fitting_results])
        else:
            mask = self.fitting_results[-1].fitter.template.mask
            transform = self.fitting_results[-1].fitter.transform
            interpolator = self.fitting_results[-1].fitter.interpolator
            warped_images = []
            for t in self.shapes():
                transform.set_target(t)
                image = self.image.warp_to(mask, transform,
                                           interpolator=interpolator)
                if as_pixels:
                    image = image.pixels

                warped_images.append(image)

        return warped_images

    def appearance_reconstructions(self, as_pixels=False):
        r"""
        Generates a list containing the appearance reconstruction obtained at
        each fitting iteration.

        Parameters
        ----------
        as_pixels : `boolean`, optional
            Whether the result is returned as a list of :map:`Image` or
            `ndarray`.

        Returns
        -------
        appearance_reconstructions : :map:`MaskedImage` or `ndarray` list
            A list containing the appearance reconstructions obtained at each
            fitting iteration.
        """
        return self._flatten_out(
            [f.appearance_reconstructions(as_pixels=as_pixels)
             for f in self.fitting_results])

    def plot_cost(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the cost evolution throughout the fitting.
        """
        title = 'Cost evolution'
        legend = self.algorithm
        x_label = 'Number of iterations'
        y_label = 'Normalized cost'
        costs = [c for cost in self.costs for c in cost]
        total_n_iters = self.n_iters + self.n_levels
        axis_limits = [0, total_n_iters, 0, max(costs)]
        return GraphPlotter(figure_id, new_figure,
                            range(0, self.n_iters+self.n_levels), costs,
                            title=title, legend=legend, x_label=x_label,
                            y_label=y_label,
                            axis_limits=axis_limits).render(**kwargs)

    def view_warped_images(self, figure_id=None, new_figure=False,
                           channels=None, from_basic_fittings=False,
                           **kwargs):
        r"""
        Displays the warped images.
        """
        pixels_list = self.warped_images(
            from_basic_fittings=from_basic_fittings, as_pixels=True)
        return self._view_images(pixels_list, figure_id=figure_id,
                                 new_figure=new_figure, channels=channels,
                                 **kwargs)

    def view_appearance_reconstructions(self, figure_id=None,
                                        new_figure=False, channels=None,
                                        **kwargs):
        r"""
        Displays the appearance reconstructions.
        """
        pixels_list = self.appearance_reconstructions(as_pixels=True)
        return self._view_images(pixels_list, figure_id=figure_id,
                                 new_figure=new_figure, channels=channels,
                                 **kwargs)

    def view_error_images(self, figure_id=None, new_figure=False,
                          channels=None, **kwargs):
        r"""
        Displays the error images.
        """
        warped_images = self.warped_images(as_pixels=True)
        appearances = self.appearance_reconstructions(as_pixels=True)
        pixels_list = [a - i for a, i in zip(appearances, warped_images)]
        return self._view_images(pixels_list, figure_id=figure_id,
                                 new_figure=new_figure, channels=channels,
                                 **kwargs)

    def _view_images(self, pixels_list, figure_id=None, new_figure=False,
                     channels=None, **kwargs):
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)

    def print_fitting_info(self):
        super(AAMMultilevelFittingResult, self).print_fitting_info()
        #print "Initial cost: {}".format(self.initial_cost)
        #print "Final cost: {}".format(self.final_cost)
