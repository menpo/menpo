from __future__ import division
import abc
import numpy as np
from copy import deepcopy
from pybug.shape import PointCloud
from pybug.transform .affine import Scale
from pybug.aam.functions import compute_error
from pybug.visualize.base import \
    MultipleImageViewer, GraphPlotter, FittingViewer, Viewable


class FittingList(list, Viewable):
    r"""
    Enhanced list of :class:`pybug.aam.fitting.Fitting` objects or the same
    :class:`pybug.aam.fitting.FittingList` objects. It implements a series of
    methods that facilitate the generation of global fitting results.

    Parameters
    -----------
    fittings: :class:`pybug.aam.fitting.Fitting` or
              :class:`pybug.aam.fitting.FittingList` lists
        A list of fitting objects.

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, fittings, error_type='me_norm'):
        super(FittingList, self).__init__(fittings)
        self.error_type = error_type

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
        if not hasattr(self, '_final_error'):
            self._final_error = np.array([f.final_error for f in self])
        return self._final_error

    @property
    def initial_error(self):
        r"""
        Returns a ndarray containing the initial error of each fitting object.
        """
        if not hasattr(self, '_initial_error'):
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
    def _final_error_dist(self):
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
    def _initial_error_dist(self):
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
    def _final_cumulative_error_dist(self):
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
        ed, x_axis = self._final_error_dist
        ced = self._cumulative_error_dist(ed)
        return ced, x_axis

    @property
    def _initial_cumulative_error_dist(self):
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
        ed, x_axis = self._initial_error_dist
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
        x_axis, y_axis = self._final_error_dist
        y_axis = [y_axis, self._initial_error_dist[1]]
        return self._plot_dist(x_axis, y_axis, title, figure_id=figure_id,
                               new_figure=new_figure, y_limit=np.max(y_axis),
                               **kwargs)

    def plot_cumulative_error_dist(self, figure_id=None, new_figure=False,
                                   **kwargs):
        r"""
        Plots the final and initial cumulative error distributions.
        """
        title = 'Cumulative Error Distribution'
        x_axis, y_axis = self._final_cumulative_error_dist
        y_axis = [y_axis, self._initial_cumulative_error_dist[1]]
        return self._plot_dist(x_axis, y_axis, title, figure_id=figure_id,
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
    image: :class:`pybug.image.masked.MaskedImage`
        The fitted image.

    fitter: :class:`pybug.aam.fitter.Fitter`
        The fitter object used to fit the image.

    basic_fittings: :class:`pybug.aam.fitting.BasicFitting` list
        A list of basic fitting objects.

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
    """

    def __init__(self, image, fitter, basic_fittings, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        self.image = deepcopy(image)
        self.fitter = fitter
        self.basic_fittings = basic_fittings
        self.affine_correction = affine_correction
        self.error_type = error_type
        self.gt_shape = gt_shape

    @abc.abstractproperty
    def scaled_levels(self):
        r"""
        Returns True if the shape results returned by the basic fittings
        must be scaled.
        """
        pass

    @property
    def algorithm(self):
        r"""
        Returns a list containing the type of algorithm used by the
        basic fitter associated to each basic fitting.
        """
        return [f.algorithm for f in self.basic_fittings]

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
        shapes: :class:`pybug.shape.PointCoulds or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """
        downscale = self.fitter.downscale
        n = self.n_levels - 1

        shapes = []
        for j, f in enumerate(self.basic_fittings):
            if not self.scaled_levels:
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
        downscale = self.fitter.downscale
        n = self.n_levels - 1

        initial_target = self.basic_fittings[0].initial_shape
        if not self.scaled_levels:
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
            axis_limits = [0, x_limit, 0, self._error_stop]
            return GraphPlotter(figure_id, new_figure, range(0, x_limit),
                                errors, title=title, legend=legend,
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

    def view_initial_target(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the initialization from which the fitting started.
        """
        image = deepcopy(self.image)
        image.landmarks['initial_shape'] = self.initial_shape
        return image.landmarks['initial_shape'].view(
            figure_id=figure_id, new_figure=new_figure, **kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the the whole fitting procedure.
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
    image: :class:`pybug.image.masked.MaskedImage`
        The fitted image.

    aam_fitter: :class:`pybug.aam.fitter.AAMFitter`
        The aam_fitter object used to fit the image.

    basic_fittings: :class:`pybug.aam.fitting.BasicFitting` list
        A list of basic fitting objects.

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
    """

    def __init__(self, image, aam_fitter, lk_fitting_list, affine_correction,
                 gt_shape=None, error_type='me_norm'):
        super(AAMFitting, self).__init__(
            image, aam_fitter, lk_fitting_list, affine_correction,
            gt_shape=gt_shape, error_type=error_type,)

    @property
    def scaled_levels(self):
        return self.fitter.scaled_reference_frames

    @property
    def residual(self):
        r"""
        Returns a list containing the type of residual used by the basic
        fitter associated to each basic fitting.
        """
        return [f.residual.type for f in self.basic_fittings]

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
        warped_images: :class:`pybug.image.masked.MaskedImage` or ndarray list
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
        appearance_reconstructions: :class:`pybug.image.masked.MaskedImage`
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


class BasicFitting(Viewable):
    r"""
    Object that holds the state of a basic fitter object before, during
    and after it has fitted a particular image.

    Parameters
    -----------
    image: :class:`pybug.image.masked.MaskedImage`
        The fitted image.

    basic_fitter: :class:`pybug.aam.fitter.BasicFitter`
        The basic_fitter object used to fit the image.

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, image, basic_fitter, gt_shape=None,
                 error_type='me_norm'):
        self.image = deepcopy(image)
        self.basic_fitter = basic_fitter
        self.error_type = error_type
        self._gt_shape = gt_shape
        self._fitted = False

    @property
    def fitted(self):
        r"""
        True if the fitting procedure has been completed. False, if not.
        """
        return self._fitted

    @property
    def algorithm(self):
        r"""
        Returns the type of algorithm used by the basic fitter.
        """
        return self.basic_fitter.type

    @property
    def error_type(self):
        r"""
        Return the type of error.
        """
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        r"""
        Sets the error type according to a set of predefined options.
        """
        if error_type is 'me_norm':
            self._error_text = 'Point-to-point error normalized by object ' \
                               'size'
        elif error_type is 'me':
            NotImplementedError("me not implemented yet")
        elif error_type is 'rmse':
            NotImplementedError("rmse not implemented yet")
        else:
            raise ValueError('Unknown error_type string selected. Valid'
                             'options are: me_norm, me, rmse')
        self._error_type = error_type

    @abc.abstractproperty
    def n_iters(self):
        r"""
        Returns the number of iterations used to fit the image.
        """
        pass

    @abc.abstractmethod
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
        shapes: :class:`pybug.shape.PointCoulds or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """
        pass

    @property
    def errors(self):
        r"""
        Returns a list containing the error at each fitting iteration.
        """
        if hasattr(self, 'gt_shape'):
            return [compute_error(t, self.gt_shape.points,
                                  self.error_type)
                    for t in self.shapes(as_points=True)]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    @abc.abstractproperty
    def final_shape(self):
        r"""
        Returns the final fitted shape.
        """
        pass

    @abc.abstractproperty
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """
        pass

    @property
    def gt_shape(self):
        r"""
        Returns the original ground truth shape associated to the image.
        """
        return self._gt_shape

    @property
    def final_error(self):
        r"""
        Returns the final fitting error.
        """
        if hasattr(self, 'ground_truth'):
            return compute_error(self.final_shape.points,
                                 self.gt_shape.points,
                                 self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    @property
    def initial_error(self):
        r"""
        Returns the initial fitting error.
        """
        if hasattr(self, 'ground_truth'):
            return compute_error(self.initial_shape.points,
                                 self.gt_shape.points,
                                 self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the error evolution throughout the fitting.
        """
        if hasattr(self, 'ground_truth'):
            legend = [self.algorithm]
            x_label = 'Number of iterations'
            y_label = self._error_text
            return GraphPlotter(figure_id, new_figure,
                                range(0, self.n_iters+1), self.errors,
                                legend=legend, x_label=x_label,
                                y_label=y_label).render(**kwargs)
        else:
            raise ValueError('Ground truth has not been set, error '
                             'cannot be plotted')

    def view_final_fitting(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the final fitting result.
        """
        image = deepcopy(self.image)
        image.landmarks['fitting'] = self.final_shape
        return image.landmarks['fitting'].view(
            figure_id=figure_id, new_figure=new_figure).render(**kwargs)

    def view_initialization(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the initialization from which the fitting started.
        """
        image = deepcopy(self.image)
        image.landmarks['fitting'] = self.initial_shape
        return image.landmarks['fitting'].view(
            figure_id=figure_id, new_figure=new_figure).render(**kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the the whole fitting procedure.
        """
        pixels = self.image.pixels
        targets = self.shapes(as_points=True)
        return FittingViewer(figure_id, new_figure, self.image.n_dims, pixels,
                             targets).render(**kwargs)


class LKFitting(BasicFitting):
    r"""
    Object that holds the state of a lucas-kanade object before, during
    and after it has fitted a particular image.

    Parameters
    -----------
    image: :class:`pybug.image.masked.MaskedImage`
        The fitted image.

    lk: :class:`pybug.aam.fitter.BasicFitter`
        The basic_fitter object used to fit the image.

    parameters: ndarray list
        A list containing the parameters of the lk object transform per
        fitting iteration.

        Default: None

    weights: ndarray list
        A list containing the parameters of the lk appearance model
        per fitting iteration. Note that, some lk objects do not explicitly
        recover the parameters of the appearance model; in this cases
        weights is None.

        Default: None

    costs: ndarray list
        A list containing the values of the cost function optimized by the
        lk object per fitting iteration.

        Default: None

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, image, lk, parameters=None, weights=None, costs=None,
                 gt_shape=None, error_type='me_norm'):
        super(LKFitting, self).__init__(image, lk, gt_shape=gt_shape,
                                        error_type=error_type)
        self.parameters = parameters
        self.weights = weights
        self.costs = costs

    @BasicFitting.fitted.setter
    def fitted(self, value):
        r"""
        Setter for the fitted property.
        """
        if value and type(value) is bool:
            if len(self.parameters) < 2 and \
               len(self.parameters): #is not len(self.cost):
                raise ValueError("Lists containing parameters and costs "
                                 "must have the same length")
            if self.weights and \
               (len(self.parameters) is not len(self.weights)):
                raise ValueError("Lists containing parameters, costs and "
                                 "weights must have the same length")
            self._fitted = value
        else:
            raise ValueError("Fitted can only be set to True")


    @property
    def residual(self):
        r"""
        Returns the type of residual used by the basic fitter.
        """
        return self.basic_fitter.residual.type

    @property
    def n_iters(self):
        return len(self.parameters) - 1

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.basic_fitter.transform.from_vector(p)
                for p in self.parameters]

    def shapes(self, as_points=False):
        if as_points:
            return [self.basic_fitter.transform.from_vector(p).target.points
                    for p in self.parameters]

        else:
            return [self.basic_fitter.transform.from_vector(p).target
                    for p in self.parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.basic_fitter.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.basic_fitter.transform.from_vector(self.parameters[0])

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target

    @property
    def final_cost(self):
        r"""
        Returns the value of the cost function at the final iteration.
        """
        return self.costs[-1]

    @property
    def initial_cost(self):
        r"""
        Returns the value of the cost function at the first iteration.
        """
        return self.costs[0]

    @BasicFitting.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if type(value) is PointCloud:
            self._gt_shape = value
        elif type(value) is list and value[0] is float:
            transform = self.basic_fitter.transform.from_vector(value)
            self._gt_shape = transform.target
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "`pybug.shape.PointClouds` or float lists"
                             "specifying transform parameters.")

    def warped_images(self, as_pixels=False):
        r"""
        Generates a list containing the warped images obtained at each fitting
        iteration.

        Parameters
        -----------
        as_pixels: boolean, optional
            Whether the result is returned as a list of Images or ndarrays.

            Default: False

        Returns
        -------
        warped_images: :class:`pybug.image.masked.MaskedImage` or ndarray list
            A list containing the warped images obtained at each fitting
            iteration.
        """
        mask = self.basic_fitter.template.mask
        transform = self.basic_fitter.transform
        interpolator = self.basic_fitter._interpolator
        if as_pixels:
            return [self.image.warp_to(mask, transform.from_vector(p),
                                       interpolator=interpolator).pixels
                    for p in self.parameters]
        else:
            return [self.image.warp_to(mask, transform.from_vector(p),
                                       interpolator=interpolator)
                    for p in self.parameters]

    def appearance_reconstructions(self, as_pixels=False):
        r"""
        Generates a list containing the appearance reconstruction obtained at
        each fitting iteration.

        Parameters
        -----------
        as_pixels: boolean, optional
            Whether the result is returned as a list of Images or ndarrays.

            Default: False

        Returns
        -------
        appearance_reconstructions: :class:`pybug.image.masked.MaskedImage`
                                    or ndarray list
            A list containing the appearance reconstructions obtained at each
            fitting iteration.
        """
        if self.weights:
            if as_pixels:
                return [self.basic_fitter.appearance_model.instance(w).pixels
                        for w in self.weights]
            else:
                return [self.basic_fitter.appearance_model.instance(w)
                        for w in self.weights]
        else:
            if as_pixels:
                return [self.basic_fitter.template.pixels for _ in self.parameters]
            else:
                return [self.basic_fitter.template for _ in self.parameters]

    def plot_cost(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the cost evolution throughout the fitting.
        """
        legend = self.algorithm
        x_label = 'Number of iterations'
        y_label = 'Normalized cost'
        return GraphPlotter(figure_id, new_figure, range(0, self.n_iters+1),
                            self.costs, legend=legend, x_label=x_label,
                            y_label=y_label).render(**kwargs)

    def view_warped_images(self, figure_id=None, new_figure=False,
                           channels=None, **kwargs):
        r"""
        Displays the warped images.
        """
        pixels_list = self.warped_images(as_pixels=True)
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)

    def view_appearance_reconstructions(self, figure_id=None,
                                        new_figure=False, channels=None,
                                        **kwargs):
        r"""
        Displays the appearance recontructions.
        """
        pixels_list = self.appearance_reconstructions(as_pixels=True)
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)

    def view_error_images(self, figure_id=None, new_figure=False,
                          channels=None, **kwargs):
        r"""
        Displays the error images.
        """
        warped_images = self.warped_images(as_pixels=True)
        appearances = self.appearance_reconstructions(as_pixels=True)
        pixels_list = [a - i for a, i in zip(appearances, warped_images)]
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)



