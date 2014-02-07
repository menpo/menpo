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

    def __init__(self, fitting_list, error_type='me_norm'):
        super(FittingList, self).__init__(fitting_list)
        self.error_type = error_type

    @property
    def algorithm(self):
        return self[0].algorithm

    @property
    def total_n_fittings(self):
        return np.product(self.final_error.shape)

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        if error_type is 'me_norm':
            for f in self:
                f.error_type = error_type
            self._error_stop = 0.1
            self._error_step = 0.001
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

    @property
    def final_error(self):
        if not hasattr(self, '_final_error'):
            self._final_error = np.array([f.final_error for f in self])
        return self._final_error

    @property
    def initial_error(self):
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
    def final_convergence(self):
        return (np.sum(self.initial_error > self.final_error) /
                self.total_n_fittings)

    @property
    def _final_error_dist(self):
        final_error = self.final_error
        return self._error_dist(final_error)

    @property
    def _initial_error_dist(self):
        initial_error = self.initial_error
        return self._error_dist(initial_error)

    def _error_dist(self, error):
        n_errors = np.product(error.shape)
        x_axis = np.arange(0, self._error_stop, self._error_step)
        y_axis = np.array([np.count_nonzero((limit-self._error_step) <
                                            error[error <= limit])
                           for limit in x_axis]) / n_errors
        return x_axis, y_axis

    @property
    def _final_cumulative_error_dist(self):
        x_axis, final_error_dist = self._final_error_dist
        y_axis = self._cumulative_error_dist(final_error_dist)
        return x_axis, y_axis

    @property
    def _initial_cumulative_error_dist(self):
        x_axis, initial_error_dist = self._initial_error_dist
        y_axis = self._cumulative_error_dist(initial_error_dist)
        return x_axis, y_axis

    @staticmethod
    def _cumulative_error_dist(error_dist):
        return np.array([np.sum(error_dist[:j])
                         for j, _ in enumerate(error_dist)])

    def plot_error_dist(self, figure_id=None, new_figure=False, labels=False,
                        **kwargs):
        title = 'Error Distribution'
        x_axis, y_axis = self._final_error_dist
        y_axis = [y_axis, self._initial_error_dist[1]]
        return self._plot_dist(x_axis, y_axis, title, figure_id=figure_id,
                               new_figure=new_figure, labels=labels, **kwargs)

    def plot_cumulative_error_dist(self, figure_id=None, new_figure=False,
                                   labels=False, **kwargs):
        title = 'Cumulative Error Distribution'
        x_axis, y_axis = self._final_cumulative_error_dist
        y_axis = [y_axis, self._initial_cumulative_error_dist[1]]
        return self._plot_dist(x_axis, y_axis, title, figure_id=figure_id,
                               new_figure=new_figure, labels=labels, **kwargs)

    def _plot_dist(self, x_axis, y_axis, title, figure_id=None,
                   new_figure=False, labels=False, **kwargs):
        legend = [self.algorithm +
                  '\nmean: {0:.4f}'.format(self.final_mean_error) +
                  'std: {0:.4f}, '.format(self.final_std_error) +
                  'median: {0:.4f}, '.format(self.final_median_error) +
                  'convergence: {0:.2f}, '.format(self.final_convergence),
                  'Initialization' +
                  '\nmean: {0:.4f}, '.format(self.initial_mean_error) +
                  'std: {0:.4f}, '.format(self.initial_std_error) +
                  'median: {0:.4f}, '.format(self.initial_median_error)]
        x_label = self._error_text
        y_label = 'Proportion of images'
        axis_limits = [0, self._error_stop, 0, 1]
        return GraphPlotter(figure_id, new_figure, x_axis, y_axis,
                            title=title, legend=legend,
                            x_label=x_label, y_label=y_label,
                            axis_limits=axis_limits).render(**kwargs)

    def view_final_target(self, figure_id=None, new_figure=False, **kwargs):
        for f in self:
            f.view_final_target(figure_id=figure_id, new_figure=new_figure,
                                **kwargs)

    def view_initial_target(self, figure_id=None, new_figure=False,
                            **kwargs):
        for f in self:
            f.view_initial_target(figure_id=figure_id, new_figure=new_figure,
                                  **kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        for f in self:
            f.view(figure_id=figure_id, new_figure=False, **kwargs)


class Fitting(Viewable):

    def __init__(self, image, fitter, basic_fitting_list, affine_correction,
                 error_type='me_norm', ground_truth=None):
        self.fitter = fitter
        self.image = deepcopy(image)
        self.basic_fitting_list = basic_fitting_list
        self.affine_correction = affine_correction
        self._error_type = error_type
        self.gt_target = ground_truth

    @property
    def algorithm(self):
        return self.basic_fitting_list[0].algorithm

    @property
    def fitted(self):
        return set([f.fitted for f in self.basic_fitting_list])

    @abc.abstractproperty
    def scaled_levels(self):
        pass

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
        if error_type is 'me_norm':
            for f in self.basic_fitting_list:
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
    def n_iters(self):
        n_iters = 0
        for f in self.basic_fitting_list:
            n_iters += f.n_iters
        return n_iters

    @property
    def n_levels(self):
        return self.fitter.n_levels

    def targets(self, as_points=False):
        downscale = self.fitter.downscale
        n = self.fitter.n_levels - 1

        targets = []
        for j, f in enumerate(self.basic_fitting_list):
            if not self.scaled_levels:
                transform = Scale(downscale**(n-j), 2)
                for t in f.targets(as_points=as_points):
                    transform.apply_inplace(t)
                    targets.append(self.affine_correction.apply(t))
            else:
                for t in f.targets(as_points=as_points):
                    targets.append(self.affine_correction.apply(t))

        return targets

    @property
    def errors(self):
        if self.gt_target is not None:
            return [compute_error(t, self.gt_target.points, self.error_type)
                    for t in self.targets(as_points=True)]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    @property
    def final_target(self):
        return self.affine_correction.apply(
            self.basic_fitting_list[-1].final_target)

    @property
    def initial_target(self):
        downscale = self.fitter.downscale
        n = self.fitter.n_levels - 1

        initial_target = self.basic_fitting_list[0].initial_target
        if not self.scaled_levels:
            Scale(downscale ** n,
                  initial_target.n_dims).apply_inplace(initial_target)

        return self.affine_correction.apply(initial_target)

    @property
    def final_error(self):
        if self.gt_target is not None:
            return compute_error(self.final_target.points,
                                 self.gt_target.points, self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    @property
    def initial_error(self):
        if self.gt_target is not None:
            return compute_error(self.initial_target.points,
                                 self.gt_target.points, self.error_type)
        else:
            raise ValueError('Ground truth has not been set, initial error '
                             'cannot be computed')

    @property
    def gt_target(self):
        return self._gt_target

    @gt_target.setter
    def gt_target(self, value):
        self._gt_target = value

    def print_final_error(self):
        print "Final error: {}".format(self.final_error)

    def print_initial_error(self):
        print "Initial error: {}".format(self.initial_error)

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
        if self.gt_target is not None:
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
            raise ValueError('Ground truth has not been set, error '
                             'cannot be plotted')

    def view_final_target(self, figure_id=None, new_figure=False, **kwargs):
        image = deepcopy(self.image)
        image.landmarks['final_target'] = self.final_target
        return image.landmarks['final_target'].view(
            figure_id=figure_id, new_figure=new_figure, **kwargs)

    def view_initial_target(self, figure_id=None, new_figure=False, **kwargs):
        image = deepcopy(self.image)
        image.landmarks['initial_target'] = self.initial_target
        return image.landmarks['initial_target'].view(
            figure_id=figure_id, new_figure=new_figure, **kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        pixels_to_view = self.image.pixels
        targets_to_view = self.targets(as_points=True)
        return FittingViewer(figure_id, new_figure,
                             self.image.n_dims, pixels_to_view,
                             targets_to_view).render(**kwargs)


class AAMFitting(Fitting):

    def __init__(self, image, aam_fitter, lk_fitting_list, affine_correction,
                 error_type='me_norm', ground_truth=None):
        super(AAMFitting, self).__init__(
            image, aam_fitter, lk_fitting_list, affine_correction,
            error_type=error_type, ground_truth=ground_truth)

    @property
    def scaled_levels(self):
        return self.fitter.scaled_reference_frames

    @property
    def residual(self):
        return self.basic_fitting_list[-1].residual.type

    @property
    def costs(self):
        return self._flatten_out([f.costs for f in self.basic_fitting_list])

    @staticmethod
    def _flatten_out(list_of_lists):
        return [i for l in list_of_lists for i in l]

    @property
    def final_cost(self):
        return self.basic_fitting_list[-1].final_cost

    @property
    def initial_cost(self):
        return self.basic_fitting_list[0].initial_cost

    def warped_images(self, from_basic_fittings=False, as_pixels=False):
        if from_basic_fittings:
            return self._flatten_out([f.warped_images(as_pixels=as_pixels)
                                      for f in self.basic_fitting_list])
        else:
            mask = self.basic_fitting_list[-1].basic_fitter.template.mask
            transform = self.basic_fitting_list[-1].basic_fitter.transform
            interpolator = \
                self.basic_fitting_list[-1].basic_fitter._interpolator
            warped_images = []
            for t in self.targets():
                transform.target = t
                image = self.image.warp_to(mask, transform,
                                           interpolator=interpolator)
                if as_pixels:
                    image = image.pixels

                warped_images.append(image)

        return warped_images

    def appearance_reconstructions(self, as_pixels=False):
        return self._flatten_out([f.appearance_reconstructions(
            as_pixels=as_pixels) for f in self.basic_fitting_list])

    def print_final_cost(self):
        print "Final cost: {}".format(self.final_cost)

    def print_initial_cost(self):
        print "Initial cost: {}".format(self.initial_cost)

    def plot_cost(self, figure_id=None, new_figure=False, **kwargs):
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

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
        if self.gt_target is not None:
            title = 'Error evolution'
            legend = [self.algorithm]
            x_label = 'Number of iterations'
            y_label = self._error_text
            errors = [e for error in self.errors for e in error]
            total_n_iters = self.n_iters + self.n_levels
            axis_limits = [0, total_n_iters, 0, self._error_stop]
            return GraphPlotter(figure_id, new_figure,
                                range(0, total_n_iters),
                                errors, title=title, legend=legend,
                                x_label=x_label, y_label=y_label,
                                axis_limits=axis_limits).render(**kwargs)
        else:
            raise ValueError('Ground truth has not been set, error '
                             'cannot be plotted')

    def view_warped_images(self, figure_id=None, new_figure=False,
                           channels=None, from_basic_fittings=False,
                           **kwargs):
        pixels_list = self.warped_images(
            from_basic_fittings=from_basic_fittings, as_pixels=True)
        return self._view_images(pixels_list, figure_id=figure_id,
                                 new_figure=new_figure, channels=channels,
                                 **kwargs)

    def view_appearance_reconstructions(self, figure_id=None,
                                        new_figure=False, channels=None,
                                        **kwargs):
        pixels_list = self.appearance_reconstructions(as_pixels=True)
        return self._view_images(pixels_list, figure_id=figure_id,
                                 new_figure=new_figure, channels=channels,
                                 **kwargs)

    def view_error_images(self, figure_id=None, new_figure=False,
                          channels=None, **kwargs):
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

    def __init__(self, lk, image, error_type='me_norm'):
        self.basic_fitter = lk
        self.image = deepcopy(image)
        self.error_type = error_type
        self._fitted = False

    @property
    def fitted(self):
        return self._fitted

    @property
    def algorithm(self):
        return self.basic_fitter.type

    @property
    def error_type(self):
        return self._error_type

    @error_type.setter
    def error_type(self, error_type):
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
        pass

    @abc.abstractmethod
    def targets(self, as_points=False):
        pass

    @property
    def errors(self):
        if hasattr(self, 'ground_truth'):
            return [compute_error(t, self.gt_target.points,
                                  self.error_type)
                    for t in self.targets(as_points=True)]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    @abc.abstractproperty
    def final_target(self):
        pass

    @abc.abstractproperty
    def initial_target(self):
        pass

    @property
    def gt_target(self):
        return self._gt_target

    @property
    def final_error(self):
        if hasattr(self, 'ground_truth'):
            return compute_error(self.final_target.points,
                                 self.gt_target.points,
                                 self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    @property
    def initial_error(self):
        if hasattr(self, 'ground_truth'):
            return compute_error(self.initial_target.points,
                                 self.gt_target.points,
                                 self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
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

    def view_final_target(self, figure_id=None, new_figure=False, **kwargs):
        image = deepcopy(self.image)
        image.landmarks['fitting'] = self.final_target
        return image.landmarks['fitting'].view(
            figure_id=figure_id, new_figure=new_figure).render(**kwargs)

    def view_initial_target(self, figure_id=None, new_figure=False, **kwargs):
        image = deepcopy(self.image)
        image.landmarks['fitting'] = self.initial_target
        return image.landmarks['fitting'].view(
            figure_id=figure_id, new_figure=new_figure).render(**kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        pixels = self.image.pixels
        targets = self.targets(as_points=True)
        return FittingViewer(figure_id, new_figure, self.image.n_dims,
                             pixels, targets).render(**kwargs)


class LKFitting(BasicFitting):

    def __init__(self, lk, image, parameters=None, weights=None, costs=None,
                 error_type='me_norm'):
        super(LKFitting, self).__init__(lk, image, error_type)
        self.parameters = parameters
        self.weights = weights
        self.costs = costs

    @BasicFitting.fitted.setter
    def fitted(self, value):
        if value and type(value) is bool:
            if len(self.parameters) < 2 and \
               len(self.parameters): #is not len(self.cost):
                raise ValueError("Lists containing parameters and costs "
                                 "must have the same length")
            if self.weights and (len(self.parameters) is not len(self.weights)):
                raise ValueError("Lists containing parameters, costs and weights "
                             "must have the same length")
            self._fitted = value
        else:
            raise ValueError("Fitted can only be set to True")


    @property
    def residual(self):
        return self.basic_fitter.residual.type

    @property
    def n_iters(self):
        return len(self.parameters) - 1

    @property
    def transforms(self):
        return [self.basic_fitter.transform.from_vector(p)
                for p in self.parameters]

    def targets(self, as_points=False):
        if as_points:
            return [self.basic_fitter.transform.from_vector(p).target.points
                    for p in self.parameters]

        else:
            return [self.basic_fitter.transform.from_vector(p).target
                    for p in self.parameters]

    @property
    def final_transform(self):
        return self.basic_fitter.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        return self.basic_fitter.transform.from_vector(self.parameters[0])

    @property
    def final_target(self):
        return self.final_transform.target

    @property
    def initial_target(self):
        return self.initial_transform.target

    @property
    def final_cost(self):
        return self.costs[-1]

    @property
    def initial_cost(self):
        return self.costs[0]

    @BasicFitting.gt_target.setter
    def gt_target(self, value):
        if type(value) is PointCloud:
            self.ground_truth = value
        elif type(value) is list and value[0] is float:
            transform = self.basic_fitter.transform.from_vector(value)
            self.ground_truth = transform.target
        else:
            raise ValueError("Accepted values for gt_target setter are "
                             "`pybug.shape.PointClouds` or float lists"
                             "specifying transform parameters.")

    def warped_images(self, as_pixels=False):
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
        legend = self.algorithm
        x_label = 'Number of iterations'
        y_label = 'Normalized cost'
        return GraphPlotter(figure_id, new_figure, range(0, self.n_iters+1),
                            self.costs, legend=legend, x_label=x_label,
                            y_label=y_label).render(**kwargs)

    def view_warped_images(self, figure_id=None, new_figure=False,
                           channels=None, **kwargs):
        pixels_list = self.warped_images(as_pixels=True)
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)

    def view_appearance_reconstructions(self, figure_id=None,
                                        new_figure=False, channels=None,
                                        **kwargs):
        pixels_list = self.appearance_reconstructions(as_pixels=True)
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)

    def view_error_images(self, figure_id=None, new_figure=False,
                          channels=None, **kwargs):
        warped_images = self.warped_images(as_pixels=True)
        appearances = self.appearance_reconstructions(as_pixels=True)
        pixels_list = [a - i for a, i in zip(appearances, warped_images)]
        return MultipleImageViewer(figure_id, new_figure,
                                   self.image.n_dims, pixels_list,
                                   channels=channels).render(**kwargs)



