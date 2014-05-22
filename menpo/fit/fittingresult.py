from __future__ import division
import abc
from copy import deepcopy
import numpy as np

from menpo.shape.pointcloud import PointCloud
from menpo.fitmultilevel.functions import compute_error
from menpo.visualize.base import (Viewable, GraphPlotter, FittingViewer,
                                  MultipleImageViewer)


class FittingResult(Viewable):
    r"""
    Object that holds the state of a :map:`Fitter` object before, during
    and after it has fitted a particular image.

    Parameters
    -----------
    image: :map:`MaskedImage`
        The fitted image.
    fitter: :map:`Fitter`
        The Fitter object used to fitter the image.
    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, image, fitter, gt_shape=None, error_type='me_norm'):
        # Initialize the error internal properties
        self.image = deepcopy(image)
        self._error_type, self._error_text = None, None
        self.fitter = fitter
        self.error_type = error_type
        self._gt_shape = gt_shape
        self._fitted = False

    @property
    def algorithm(self):
        r"""
        Returns the name of the algorithm used by the Fitter.
        """
        return self.fitter.algorithm

    @property
    def fitted(self):
        r"""
        True iff the fitting procedure has been completed.
        """
        return self._fitted

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
        if error_type == 'me_norm':
            self._error_text = ('Point-to-point error normalized by object '
                                'size')
        elif error_type == 'me':
            raise NotImplementedError("me not implemented yet")
        elif error_type == 'rmse':
            raise NotImplementedError("rmse not implemented yet")
        else:
            raise ValueError('Unknown error_type string selected. Valid'
                             'options are: me_norm, me, rmse')
        self._error_type = error_type

    @abc.abstractproperty
    def n_iters(self):
        r"""
        Returns the number of iterations used to fit the image.
        """

    @abc.abstractmethod
    def shapes(self, as_points=False):
        r"""
        Generates a list containing the shapes obtained at each fitting
        iteration.

        Parameters
        -----------
        as_points : boolean, optional
            Whether the results is returned as a list of :map:`PointCloud`s or
            ndarrays.

            Default: `False`

        Returns
        -------
        shapes : :map:`PointCloud`s or ndarray list
            A list containing the shapes obtained at each fitting iteration.
        """

    @property
    def errors(self):
        r"""
        Returns a list containing the error at each fitting iteration.
        """
        if self.gt_shape is not None:
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

    @abc.abstractproperty
    def initial_shape(self):
        r"""
        Returns the initial shape from which the fitting started.
        """

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

        :type: float
        """
        if self.gt_shape is not None:
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

        :type: float
        """
        if self.gt_shape is not None:
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
        if self.gt_shape is not None:
            legend = [self.algorithm]
            x_label = 'Number of iterations'
            y_label = self._error_text
            return GraphPlotter(figure_id, new_figure,
                                range(0, self.n_iters+1), [self.errors],
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

    def view_ground_truth(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the ground truth annotation.
        """
        if self.gt_shape is not None:
            image = deepcopy(self.image)
            image.landmarks['gt_shape'] = self.gt_shape
            return image.landmarks['gt_shape'].view(
                figure_id=figure_id, new_figure=new_figure, **kwargs)
        else:
            raise ValueError('Ground truth shape has not been set.')

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the whole fitting procedure.
        """
        pixels = self.image.pixels
        targets = self.shapes(as_points=True)
        return FittingViewer(figure_id, new_figure, self.image.n_dims, pixels,
                             targets).render(**kwargs)


# TODO: document me
class NonParametricFittingResult(FittingResult):
    r"""
    """

    def __init__(self, image, fitter, shapes=None, costs=None,
                 gt_shape=None, error_type='me_norm'):
        super(NonParametricFittingResult, self).__init__(
            image, fitter, gt_shape=gt_shape, error_type=error_type)
        self.parameters = shapes
        self.costs = costs

    @FittingResult.fitted.setter
    def fitted(self, value):
        r"""
        Setter for the fitted property.
        """
        if value and type(value) is bool:
            if len(self.parameters) < 2:
               # and len(self.parameters) is not len(self.cost):
                raise ValueError("Lists containing weights and costs "
                                 "must have the same length")
            self._fitted = value
        else:
            raise ValueError("Fitted can only be set to True")

    @property
    def n_iters(self):
        return len(self.shapes()) - 1

    def shapes(self, as_points=False):
        if as_points:
            return [deepcopy(s.points) for s in self.parameters]

        else:
            return deepcopy(self.parameters)

    @property
    def final_shape(self):
        return deepcopy(self.parameters[-1])

    @property
    def initial_shape(self):
        return deepcopy(self.parameters[0])

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if type(value) is PointCloud:
            self._gt_shape = value
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "`menpo.shape.PointClouds`.")

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

    def plot_cost(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Plots the cost evolution throughout the fitting.
        """
        legend = self.algorithm
        x_label = 'Number of iterations'
        y_label = 'Normalized cost'
        return GraphPlotter(figure_id, new_figure, range(0, self.n_iters+1),
                            [self.costs], legend=legend, x_label=x_label,
                            y_label=y_label).render(**kwargs)


#TODO: Document me
class SemiParametricFittingResult(NonParametricFittingResult):
    r"""
    """
    def __init__(self, image, fitter, parameters=None, costs=None,
                 gt_shape=None, error_type='me_norm'):
        super(SemiParametricFittingResult, self).__init__(
            image, fitter, gt_shape=gt_shape, error_type=error_type)
        self.parameters = parameters
        self.costs = costs

    @FittingResult.fitted.setter
    def fitted(self, value):
        r"""
        Setter for the fitted property.
        """
        if value and type(value) is bool:
            if len(self.parameters) < 2:
               # and len(self.parameters) is not len(self.cost):
                raise ValueError("Lists containing weights and costs "
                                 "must have the same length")
            self._fitted = value
        else:
            raise ValueError("Fitted can only be set to True")

    @property
    def residual(self):
        r"""
        Returns the type of residual used by the basic fitter.
        """
        return self.fitter.residual.type

    @property
    def transforms(self):
        r"""
        Generates a list containing the transforms obtained at each fitting
        iteration.
        """
        return [self.fitter.transform.from_vector(p) for p in self.parameters]

    def shapes(self, as_points=False):
        if as_points:
            return [self.fitter.transform.from_vector(p).target.points
                    for p in self.parameters]

        else:
            return [self.fitter.transform.from_vector(p).target
                    for p in self.parameters]

    @property
    def final_transform(self):
        r"""
        Returns the final transform.
        """
        return self.fitter.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        r"""
        Returns the initial transform from which the fitting started.
        """
        return self.fitter.transform.from_vector(self.parameters[0])

    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target

    @FittingResult.gt_shape.setter
    def gt_shape(self, value):
        r"""
        Setter for the ground truth shape associated to the image.
        """
        if type(value) is PointCloud:
            self._gt_shape = value
        elif type(value) is list and value[0] is float:
            transform = self.fitter.transform.from_vector(value)
            self._gt_shape = transform.target
        else:
            raise ValueError("Accepted values for gt_shape setter are "
                             "`menpo.shape.PointClouds` or float lists"
                             "specifying transform parameters.")


#TODO: Document me
class ParametricFittingResult(SemiParametricFittingResult):
    r"""
    Object that holds the state of a lucas-kanade object before, during
    and after it has fitted a particular image.

    Parameters
    -----------
    image: :class:`menpo.image.masked.MaskedImage`
        The fitted image.

    lk: :class:`menpo.aam.fitter.BasicFitter`
        The fitter object used to fitter the image.

    weights: ndarray list
        A list containing the weights of the lk object transform per
        fitting iteration.

        Default: None

    weights: ndarray list
        A list containing the weights of the lk appearance model
        per fitting iteration. Note that, some lk objects do not explicitly
        recover the weights of the appearance model; in this cases
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
        super(ParametricFittingResult, self).__init__(
            image, lk, gt_shape=gt_shape, error_type=error_type)
        self.parameters = parameters
        self.weights = weights
        self.costs = costs

    @FittingResult.fitted.setter
    def fitted(self, value):
        r"""
        Setter for the fitted property.
        """
        if value and type(value) is bool:
            if len(self.parameters) < 2 and \
               len(self.parameters): #is not len(self.cost):
                raise ValueError("Lists containing weights and costs "
                                 "must have the same length")
            if self.weights and \
               (len(self.parameters) is not len(self.weights)):
                raise ValueError("Lists containing weights, costs and "
                                 "weights must have the same length")
            self._fitted = value
        else:
            raise ValueError("Fitted can only be set to True")

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
        warped_images: :class:`menpo.image.masked.MaskedImage` or ndarray list
            A list containing the warped images obtained at each fitting
            iteration.
        """
        mask = self.fitter.template.mask
        transform = self.fitter.transform
        interpolator = self.fitter.interpolator
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
        appearance_reconstructions: :class:`menpo.image.masked.MaskedImage`
                                    or ndarray list
            A list containing the appearance reconstructions obtained at each
            fitting iteration.
        """
        if self.weights:
            if as_pixels:
                return [self.fitter.appearance_model.instance(w).pixels
                        for w in self.weights]
            else:
                return [self.fitter.appearance_model.instance(w)
                        for w in self.weights]
        else:
            if as_pixels:
                return [self.fitter.template.pixels for _ in self.parameters]
            else:
                return [self.fitter.template for _ in self.parameters]

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


class FittingResultList(list, Viewable):
    r"""
    Enhanced list of :class:`menpo.fitter.fitting.FittingResults` objects. It
    implements a series of methods that facilitate the generation of global
    fitting results.

    Parameters
    -----------
    fitting_results: :class:`menpo.fitter.fitting.FittingResults`
        A list of FittingResult objects.

    error_type: 'me_norm', 'me' or 'rmse', optional.
        Specifies the way in which the error between the fitted and
        ground truth shapes is to be computed.

        Default: 'me_norm'
    """

    def __init__(self, fitting_results, error_type='me_norm'):
        super(FittingResultList, self).__init__(fitting_results)
        self.error_type = error_type
        self._final_error = None
        self._initial_error = None

    @property
    def algorithm(self):
        r"""
        Returns the name of the fitting algorithm used by the fitter object
        associated to the fitting objects.
        """
        # TODO: ensure that all basic_fitting algorithms are the same?
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
        if error_type == 'me_norm':
            for f in self:
                f.error_type = error_type
            self._error_stop = 0.1
            self._error_step = 0.001
            self._error_text = 'Point-to-point error normalized by object ' \
                               'size'
        elif error_type == 'me':
            NotImplementedError('me not implemented yet')
        elif error_type == 'rmse':
            NotImplementedError('rmse not implemented yet')
        else:
            raise ValueError('Unknown error_type string selected. Valid'
                             'options are: "me_norm", "me", "rmse".')
        self._error_type = error_type

    @property
    def n_fittings(self):
        r"""
        Returns the total number of fitting, i.e. the length of the list.
        """
        return len(self)

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
        ed = np.array([np.count_nonzero((limit-self._error_step) <=
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
        raise ValueError("Not implemented yet")

    def view_initialization(self, figure_id=None, new_figure=False,
                            **kwargs):
        r"""
        Displays the initialization used by each fitting object.
        """
        raise ValueError("Not implemented yet")

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the the whole fitting procedure for each fitting object.
        """
        raise ValueError("Not implemented yet")


class TrackingResultList(FittingResultList):

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        r"""
        Displays the final tracking result for the whole sequence.
        """
        raise ValueError("Not implemented yet")
