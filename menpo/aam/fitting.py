from __future__ import division
import numpy as np
from copy import deepcopy
from menpo.transform import Scale
from menpo.aam.functions import compute_error
from menpo.visualize.base import (MultipleImageViewer, GraphPlotter,
                                  FittingViewer, Viewable)


class FittingList(list, Viewable):
    r"""
    Enhanced list of :class:`menpo.aam.fitting.Fitting` objects or the same
    :class:`menpo.aam.fitting.FittingList` objects. It implements a series of
    methods that facilitate the generation of global fitting results.

    Parameters
    -----------
    fittings: :class:`menpo.aam.fitting.Fitting` or
              :class:`menpo.aam.fitting.FittingList` lists
        A list of fitting objects.

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, fittings, error_type='me_norm'):
        super(FittingList, self).__init__(fittings)
        self.error_type = error_type
        self._final_error = None
        self._initial_error = None

    @property
    def algorithm(self):
        r"""
        Returns the type of algorithm used by the fitter object associated
        to the fitting objects.
        """
        return self[0].algorithm

    @property
    def error_type(self):
        r"""
        Returns the type of error.
        """
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        r"""
        Sets the error type according to a set of predefined options.
        """
        if error_type is 'me_norm':
            for f in self:
                f.error_type = error_type
            self._error_stop = 0.1
            self._error_step = 0.001
            self._error_text = 'Point-to-point error normalized by object ' \
                               'size'
        elif error_type is 'me':
            NotImplementedError('me not implemented yet')
        elif error_type is 'rmse':
            NotImplementedError('rmse not implemented yet')
        else:
            raise ValueError('Unknown error_type string selected. Valid'
                             'options are: "me_norm", "me", "rmse".')
        self._error_type = error_type

    @property
    def n_fittings(self):
        r"""
        Returns the total number of fitting. Note that this might not be
        equal to the total number of element in the list since we can have
        FittingList of FittingList objects.
        """
        return np.product(self.final_error.shape)

    @property
    def final_error(self):
        r"""
        Returns a ndarray containing the final error of each fitting object.
        """
        if self._final_error is None:
            self._final_error = np.array([f.final_error for f in self])
        return self._final_error

    @property
    def initial_error(self):
        r"""
        Returns a ndarray containing the initial error of each fitting object.
        """
        if self._initial_error is None:
            self._initial_error = np.array([f.initial_error for f in self])
        return self._initial_error

    @property
    def final_mean_error(self):
        return np.mean(self.final_error)

    @property
    def initial_mean_error(self):
        return np.mean(self.initial_error)

    @property
    def final_std_error(self):
        return np.std(self.final_error)

    @property
    def initial_std_error(self):
        return np.std(self.initial_error)

    @property
    def final_var_error(self):
        return np.var(self.final_error)

    @property
    def initial_var_error(self):
        return np.var(self.initial_error)

    @property
    def final_median_error(self):
        return np.median(self.final_error)

    @property
    def initial_median_error(self):
        return np.median(self.initial_error)

    @property
    def convergence(self):
        r"""
        Returns the percentage of fitting objects that converged. A fitting
        object is considered to have converged if the final fitting error is
        smaller than the initial one.
        """
        return (np.sum(self.initial_error > self.final_error) /
                self.n_fittings)

    @property
    def final_error_dist(self):
        r"""
        Computes the final error distribution among all fitting objects.

        Returns
        -------
        ed: ndarray
            The final error distribution among all fitting objects within
            the interval [0, self._error_stop]
        x_axis: ndarray
            The interval [0, self._error_stop]
        """
        final_error = self.final_error
        return self._error_dist(final_error)

    @property
    def initial_error_dist(self):
        r"""
        Computes the initial error distribution among all fitting objects.

        Returns
        -------
        ed: ndarray
            The initial error distribution among all fitting objects within
            the interval [0, self._error_stop]
        x_axis: ndarray
            The interval [0, self._error_stop]
        """
        initial_error = self.initial_error
        return self._error_dist(initial_error)

    def _error_dist(self, error):
        n_errors = np.product(error.shape)
        x_axis = np.arange(0, self._error_stop, self._error_step)
        ed = np.array([np.count_nonzero((limit-self._error_step) <
                                        error[error <= limit])
                       for limit in x_axis]) / n_errors
        return ed, x_axis

    @property
    def final_cumulative_error_dist(self):
        r"""
        Computes the final cumulative error distribution among all fitting
        objects.

        Returns
        -------
        ced: ndarray
            The final cumulative error distribution among all fitting objects
            within the interval [0, self._error_stop]
        x_axis: ndarray
            The interval [0, self._error_stop]
        """
        ed, x_axis = self.final_error_dist
        ced = self._cumulative_error_dist(ed)
        return ced, x_axis

    @property
    def initial_cumulative_error_dist(self):
        r"""
        Computes the initial cumulative error distribution among all fitting
        objects.

        Returns
        -------
        ced: ndarray
            The initial cumulative error distribution among all fitting
            objects within the interval [0, self._error_stop]
        x_axis: ndarray
            The interval [0, self._error_stop]
        """
        ed, x_axis = self.initial_error_dist
        ced = self._cumulative_error_dist(ed)
        return ced, x_axis

    @staticmethod
    def _cumulative_error_dist(error_dist):
        return np.array([np.sum(error_dist[:j])
                         for j, _ in enumerate(error_dist)])

    def plot_error_dist(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the final and initial error distributions.
        """
        title = 'Error Distribution'
        ed, x_axis = self.final_error_dist
        ed = [ed, self.initial_error_dist[0]]
        return self._plot_dist(x_axis, ed, title, figure_id=figure_id,
                               new_figure=new_figure, y_limit=np.max(ed),
                               **kwargs)

    def plot_cumulative_error_dist(self, figure_id=None, new_figure=False,
                                   **kwargs):
        r"""
        Plots the final and initial cumulative error distributions.
        """
        title = 'Cumulative Error Distribution'
        ced, x_axis = self.final_cumulative_error_dist
        ced = [ced, self.initial_cumulative_error_dist[0]]
        return self._plot_dist(x_axis, ced, title, figure_id=figure_id,
                               new_figure=new_figure, **kwargs)

    def _plot_dist(self, x_axis, y_axis, title, figure_id=None,
                   new_figure=False, y_limit=1, **kwargs):
        legend = [self.algorithm +
                  '\nmean: {0:.4f}'.format(self.final_mean_error) +
                  'std: {0:.4f}, '.format(self.final_std_error) +
                  'median: {0:.4f}, '.format(self.final_median_error) +
                  'convergence: {0:.2f}, '.format(self.convergence),
                  'Initialization' +
                  '\nmean: {0:.4f}, '.format(self.initial_mean_error) +
                  'std: {0:.4f}, '.format(self.initial_std_error) +
                  'median: {0:.4f}, '.format(self.initial_median_error)]
        x_label = self._error_text
        y_label = 'Proportion of images'
        axis_limits = [0, self._error_stop, 0, y_limit]
        return GraphPlotter(figure_id, new_figure, x_axis, y_axis,
                            title=title, legend=legend,
                            x_label=x_label, y_label=y_label,
                            axis_limits=axis_limits).render(**kwargs)

    def view_final_fitting(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the final fitting result obtained by each fitting object.
        """
        for f in self:
            f.view_final_fitting(figure_id=figure_id, new_figure=new_figure,
                                 **kwargs)

    def view_initialization(self, figure_id=None, new_figure=False,
                            **kwargs):
        r"""
        Displays the initialization used by each fitting object.
        """
        for f in self:
            f.view_initialization(figure_id=figure_id, new_figure=new_figure,
                                  **kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the the whole fitting procedure for each fitting object.
        """
        for f in self:
            f.view(figure_id=figure_id, new_figure=False, **kwargs)


class Fitting(Viewable):
    r"""
    Object that holds the state of a Fitter object (to which it is linked)
    after fitting a particular image.

    Parameters
    -----------
    image: :class:`menpo.image.masked.MaskedImage`
        The fitted image.

    fitter: :class:`menpo.aam.fitter.Fitter`
        The fitter object used to fit the image.

    basic_fittings: :class:`menpo.aam.fitting.BasicFitting` list
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

    def __init__(self, image, fitter, basic_fittings, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        self.image = deepcopy(image)
        self.fitter = fitter
        self.basic_fittings = basic_fittings
        self.affine_correction = affine_correction
        self.error_type = error_type
        self.gt_shape = gt_shape

    @property
    def scaled_levels(self):
        r"""
        Returns True if the shape results returned by the basic fittings
        must be scaled.
        """
        return self.fitter.scaled_levels

    @property
    def algorithm(self):
        r"""
        Returns the type of algorithm used by the basic fitter
        associated to each basic fitting.
        """
        # TODO: ensure that all basic_fitting algorithms are the same?
        return self.basic_fittings[0].algorithm

    @property
    def error_type(self):
        r"""
        Returns the type of error.
        """
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        r"""
        Sets the error type according to a set of predefined options.
        """
        if error_type is 'me_norm':
            for f in self.basic_fittings:
                f.error_type = error_type
            self._error_stop = 0.1
            self._error_text = 'Point-to-point error normalized by object ' \
                               'size'
        elif error_type is 'me':
            NotImplementedError("erro_type 'me' not implemented yet")
        elif error_type is 'rmse':
            NotImplementedError("error_type 'rmse' not implemented yet")
        else:
            raise ValueError("Unknown error_type string selected. Valid"
                             "options are: 'me_norm', 'me', 'rmse'")
        self._error_type = error_type

    @property
    def fitted(self):
        r"""
        Returns the fitted state of each basic fitting.
        """
        return [f.fitted for f in self.basic_fittings]

    @property
    def n_iters(self):
        r"""
        Returns the total number of iterations used to fit the image.
        """
        n_iters = 0
        for f in self.basic_fittings:
            n_iters += f.n_iters
        return n_iters

    @property
    def n_levels(self):
        r"""
        Returns the total number of basic fittings.
        """
        return len(self.basic_fittings)

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
        downscale = self.fitter._downscale
        n = self.n_levels - 1

        shapes = []
        for j, f in enumerate(self.basic_fittings):
            if downscale and not self.scaled_levels:
                transform = Scale(downscale**(n-j), 2)
                for t in f.shapes(as_points=as_points):
                    transform.apply_inplace(t)
                    shapes.append(self.affine_correction.apply(t))
            else:
                for t in f.shapes(as_points=as_points):
                    shapes.append(self.affine_correction.apply(t))

        return shapes

    @property
    def errors(self):
        r"""
        Returns a list containing the error at each fitting iteration.
        """
        if self.gt_shape is not None:
            return [compute_error(t, self.gt_shape.points, self.error_type)
                    for t in self.shapes(as_points=True)]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    @property
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """
        return self.affine_correction.apply(
            self.basic_fittings[-1].final_shape)

    @property
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """
        downscale = self.fitter._downscale
        n = self.n_levels - 1

        initial_target = self.basic_fittings[0].initial_shape
        if downscale and not self.scaled_levels:
            Scale(downscale ** n,
                  initial_target.n_dims).apply_inplace(initial_target)

        return self.affine_correction.apply(initial_target)

    @property
    def final_error(self):
        r"""
        Returns the final fitting error.
        """
        if self.gt_shape is not None:
            return compute_error(self.final_shape.points,
                                 self.gt_shape.points, self.error_type)
        else:
            raise ValueError('Ground truth shape has not been set, '
                             'final error cannot be computed')

    @property
    def initial_error(self):
        r"""
        Returns the initial error before fitting.
        """
        if self.gt_shape is not None:
            return compute_error(self.initial_shape.points,
                                 self.gt_shape.points, self.error_type)
        else:
            raise ValueError('Ground truth shape has not been set, initial '
                             'error cannot be computed')

    @property
    def gt_shape(self):
        r"""
        Returns the ground truth shape associated to the image.
        """
        return self._gt_target

    @gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        self._gt_target = value

    def print_fitting_info(self):
        r"""
        Prints information related to the fitting.
        """
        print "Initial error: {}".format(self.initial_error)
        print "Final error: {}".format(self.final_error)

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

    def view_final_fitting(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the final fitting result.
        """
        image = deepcopy(self.image)
        image.landmarks['final_shape'] = self.final_shape
        return image.landmarks['final_shape'].view(
            figure_id=figure_id, new_figure=new_figure, **kwargs)

    def view_initialization(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the initialization from which the fitting started.
        """
        image = deepcopy(self.image)
        image.landmarks['initial_shape'] = self.initial_shape
        return image.landmarks['initial_shape'].view(
            figure_id=figure_id, new_figure=new_figure, **kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the whole fitting procedure.
        """
        pixels = self.image.pixels
        targets = self.shapes(as_points=True)
        return FittingViewer(figure_id, new_figure, self.image.n_dims, pixels,
                             targets).render(**kwargs)


class AAMFitting(Fitting):
    r"""
    Object that holds the state of a Fitter object associated to an AAM
    object after this has fitted a particular image.

    Parameters
    -----------
    image: :class:`menpo.image.masked.MaskedImage`
        The fitted image.

    aam_fitter: :class:`menpo.aam.fitter.AAMFitter`
        The aam_fitter object used to fit the image.

    basic_fittings: :class:`menpo.aam.fitting.BasicFitting` list
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

    def __init__(self, image, aam_fitter, lk_fitting_list, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        super(AAMFitting, self).__init__(
            image, aam_fitter, lk_fitting_list, affine_correction,
            gt_shape=gt_shape, error_type=error_type,)

    @property
    def residual(self):
        r"""
        Returns the type of residual used by the basic fitter associated to
        each basic fitting.
        """
        # TODO: ensure that all basic_fitting residuals are the same?
        return self.basic_fittings[-1].residual.type

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
        return self.basic_fittings[-1].final_cost

    @property
    def initial_cost(self):
        r"""
        Returns the initial fitting cost.
        """
        return self.basic_fittings[0].initial_cost

    def warped_images(self, from_basic_fittings=False, as_pixels=False):
        r"""
        Generates a list containing the warped images obtained at each fitting
        iteration.

        Parameters
        -----------
        from_basic_fittings: boolean, optional
            If True, the returned transform per iteration is used to warp
            the internal image representation used by each basic fitter.
            If False, the transforms are used to warp original image.

            Default: False

        as_pixels: boolean, optional
            Whether the result is returned as a list of Images or
            ndarrays.

            Default: False

        Returns
        -------
        warped_images: :class:`menpo.image.masked.MaskedImage` or ndarray list
            A list containing the warped images obtained at each fitting
            iteration.
        """
        if from_basic_fittings:
            return self._flatten_out([f.warped_images(as_pixels=as_pixels)
                                      for f in self.basic_fittings])
        else:
            mask = self.basic_fittings[-1].basic_fitter.template.mask
            transform = self.basic_fittings[-1].basic_fitter.transform
            interpolator = \
                self.basic_fittings[-1].basic_fitter._interpolator
            warped_images = []
            for t in self.shapes():
                transform.target = t
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
        -----------
        as_pixels: boolean, optional
            Whether the result is returned as a list of Images or
            ndarrays.

            Default: False

        Returns
        -------
        appearance_reconstructions: :class:`menpo.image.masked.MaskedImage`
                                    or ndarray list
            A list containing the appearance reconstructions obtained at each
            fitting iteration.
        """
        return self._flatten_out(
            [f.appearance_reconstructions(as_pixels=as_pixels)
             for f in self.basic_fittings])

    def print_fitting_info(self):
        super(AAMFitting, self).print_fitting_info()
        #print "Initial cost: {}".format(self.initial_cost)
        #print "Final cost: {}".format(self.final_cost)

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
