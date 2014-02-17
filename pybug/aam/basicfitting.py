from __future__ import division
import abc
from copy import deepcopy
from pybug.shape import PointCloud
from pybug.aam.functions import compute_error
from pybug.visualize.base import \
    MultipleImageViewer, GraphPlotter, FittingViewer, Viewable


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
                            [self.costs], legend=legend, x_label=x_label,
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

