import abc
import numpy as np
from copy import deepcopy
from scipy.linalg import solve
from pybug.aam.functions import compute_error
from pybug.visualize.base import\
    MultipleImageViewer, GraphPlotter, FittingViewer, Viewable


class LucasKanade(object):
    r"""
    An abstract base class for implementations of Lucas-Kanade [1]_
    type algorithms.

    This is to abstract away optimisation specific functionality such as the
    calculation of the Hessian (which could be derived using a number of
    techniques, including Gauss-Newton and Levenberg-Marquardt).

    Parameters
    ----------
    image : :class:`pybug.image.base.Image`
        The image to perform the alignment upon.

        .. note:: Only the image is expected within the base class because
            different algorithms expect different kinds of template
            (image/model)
    residual : :class:`pybug.lucaskanade.residual.Residual`
        The kind of residual to be calculated. This is used to quantify the
        error between the input image and the reference object.
    transform : :class:`pybug.transform.base.AlignableTransform`
        The transformation type used to warp the image in to the appropriate
        reference frame. This is used by the warping function to calculate
        sub-pixel coordinates of the input image in the reference frame.
    warp : function
        A function that takes 3 arguments,
        ``warp(`` :class:`image <pybug.image.base.Image>`,
        :class:`template <pybug.image.base.Image>`,
        :class:`transform <pybug.transform.base.AlignableTransform>` ``)``
        This function is intended to perform sub-pixel interpolation of the
        pixel locations calculated by transforming the given image into the
        reference frame of the template. Appropriate functions are given in
        :doc:`pybug.interpolation`.
    optimisation : ('GN',) | ('LM', float), optional
        The optimisation technique used to calculate the Hessian approximation.
        Note that for 'LM' the float is used to set the update step.

        Default: 'GN'
    update_step : float, optional
        The update step used when performing a Levenberg-Marquardt
        optimisation.

        Default: 0.001
    eps : float, optional
        The convergence value. When calculating the level of convergence, if
        the norm of the delta parameter updates is less than ``eps``, the
        algorithm is considered to have converged.

        Default: 1**-10

    Notes
    -----
    The type of optimisation technique chosen will determine properties such
    as the convergence rate of the algorithm. The supported optimisation
    techniques are detailed below:

    ===== ==================== ===============================================
    type  full name            hessian approximation
    ===== ==================== ===============================================
    'GN'  Gauss-Newton         :math:`\mathbf{J^T J}`
    'LM'  Levenberg-Marquardt  :math:`\mathbf{J^T J + \lambda\, diag(J^T J)}`
    ===== ==================== ===============================================

    Attributes
    ----------
    transform
    parameters
    n_iters

    References
    ----------
    .. [1] Lucas, Bruce D., and Takeo Kanade.
       "An iterative image registration technique with an application to
       stereo vision." IJCAI. Vol. 81. 1981.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, residual, transform,
                 interpolator='scipy', optimisation=('GN',), eps=1**-10):
        # set basic state for all Lucas Kanade algorithms
        self.transform = transform
        self.residual = residual
        self.eps = eps

        # select the optimisation approach and warp function
        self._calculate_delta_p = self._select_optimisation(optimisation)
        self._interpolator = interpolator

    def _select_optimisation(self, optimisation):
        if optimisation[0] == 'GD':
            self.update_step = optimisation[1]
            self.__e_lm = 0
            return self._gradient_descent
        if optimisation[0] == 'GN':
            return self._gauss_newton_update
        elif optimisation[0] == 'GN_lp':
            self.lp = optimisation[1]
            return self._gauss_newton_lp_update
        elif optimisation[0] == 'LM':
            self.update_step = optimisation[1]
            self.__e_lm = 0
            return self._levenberg_marquardt_update
        else:
            raise ValueError('Unknown optimisation string selected. Valid'
                             'options are: GN, GN_lp, LM')

    def _gradient_descent(self, sd_delta_p):
        raise NotImplementedError("Gradient descent optimization not "
                                  "implemented yet")

    def _gauss_newton_update(self, sd_delta_p):
        return solve(self._H, sd_delta_p)

    def _gauss_newton_lp_update(self, sd_delta_p):
        raise NotImplementedError("Gauss-Newton lp-norm optimization not "
                                  "implemented yet")

    def _levenberg_marquardt_update(self, sd_delta_p):
        LM = np.diagflat(np.diagonal(self._H))
        H_lm = self._H + (self.update_step * LM)

        if self.residual.error < self.__e_lm:
            # Bad step, increase step
            self.update_step *= 10
        else:
            # Good step, decrease step
            self.update_step /= 10
            self.__e_lm = self.residual.error

        return solve(H_lm, sd_delta_p)

    def _precompute(self):
        """
        Performs pre-computations related to specific alignment algorithms
        """
        pass

    def align(self, image, parameters, max_iters=20, **kwargs):
        r"""
        Perform an alignment using the Lukas-Kanade framework.

        Parameters
        ----------
        max_iters : int
            The maximum number of iterations that will be used in performing
            the alignment

        Returns
        -------
        transform : :class:`pybug.transform.base.AlignableTransform`
            The final transform that optimally aligns the source to the
            target.
        """
        self.transform.from_vector_inplace(parameters)
        lk_fitting = LKFitting(self, image, parameters=[parameters])
        return self._align(lk_fitting, max_iters=max_iters, **kwargs)

    @abc.abstractmethod
    def _align(self, **kwargs):
        r"""
        Abstract method to be overridden by subclasses that implements the
        alignment algorithm.
        """
        pass


class LKFitting(Viewable):

    def __init__(self, lk, image, parameters=None, weights=None, costs=None,
                 error_type='me_norm', fitted=False):
        # self._valid_lists(parameters, costs)
        self.lk = lk
        self.image = deepcopy(image)
        self.parameters = parameters
        self.weights = weights
        self.costs = costs
        self.fitted = fitted
        self.error_type = error_type

    # @staticmethod
    # def _valid_lists(parameters, cost):
    #     if not (len(parameters) is len(cost)):
    #         raise ValueError("Lists containing parameters and costs "
    #                          "must contain the same number of elements")

    @property
    def algorithm_type(self):
        return self.lk.type

    @property
    def residual_type(self):
        return self.lk.residual.type

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

    @property
    def n_iters(self):
        return len(self.parameters) - 1

    @property
    def transforms(self):
        return [self.lk.transform.from_vector(p) for p in self.parameters]

    def targets(self, as_points=False):
        if as_points:
            return [self.lk.transform.from_vector(p).target.points
                    for p in self.parameters]

        else:
            return [self.lk.transform.from_vector(p).target
                    for p in self.parameters]

    @property
    def errors(self):
        if hasattr(self, 'ground_truth'):
            return [compute_error(t, self.ground_truth.points,
                                  self.error_type)
                    for t in self.targets(as_points=True)]
        else:
            raise ValueError('Ground truth has not been set, errors cannot '
                             'be computed')

    @property
    def final_transform(self):
        return self.lk.transform.from_vector(self.parameters[-1])

    @property
    def initial_transform(self):
        return self.lk.transform.from_vector(self.parameters[0])

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
    def final_error(self):
        if hasattr(self, 'ground_truth'):
            return compute_error(self.final_target.points,
                                 self.ground_truth.points,
                                 self.error_type)
        else:
            raise ValueError('Ground truth has not been set, final error '
                             'cannot be computed')

    def set_ground_truth(self, ground_truth, as_target=True):
        if as_target:
            self.ground_truth = ground_truth
        else:
            transform = self.lk.transform.from_vector(ground_truth)
            self.ground_truth = transform.target

    def warped_images(self, as_pixels=False):
        mask = self.lk.template.mask
        transform = self.lk.transform
        interpolator = self.lk._interpolator
        if as_pixels:
            return [self.image.warp_to(mask, transform.from_vector(p),
                                       interpolator=interpolator).pixels
                    for p in self.parameters]
        else:
            return [self.image.warp_to(mask, transform.from_vector(p),
                                       interpolator=interpolator)
                    for p in self.parameters]

    def appearances(self, as_pixels=False):
        if self.weights:
            if as_pixels:
                return [self.lk.appearance_model.instance(w).pixels
                        for w in self.weights]
            else:
                return [self.lk.appearance_model.instance(w)
                        for w in self.weights]
        else:
            if as_pixels:
                return [self.lk.template.pixels for _ in self.parameters]
            else:
                return [self.lk.template for _ in self.parameters]

    def plot_cost(self, figure_id=None, new_figure=False, **kwargs):
        legend = self.algorithm_type
        x_label = 'Number of iterations'
        y_label = 'Normalized cost'
        return GraphPlotter(figure_id, new_figure, range(0, self.n_iters+1),
                            self.costs, legend=legend, x_label=x_label,
                            y_label=y_label).render(**kwargs)

    def plot_error(self, figure_id=None, new_figure=False, **kwargs):
        if hasattr(self, 'ground_truth'):
            legend = [self.algorithm_type]
            x_label = 'Number of iterations'
            y_label = self._error_text
            return GraphPlotter(figure_id, new_figure,
                                range(0, self.n_iters+1), self.errors,
                                legend=legend, x_label=x_label,
                                y_label=y_label).render(**kwargs)
        else:
            raise ValueError('Ground truth has not been set, error '
                             'cannot be plotted')

    def view_warped_images(self, figure_id=None, new_figure=False,
                           channels=None, masked=True, **kwargs):
        pixels_list = self.warped_images(as_pixels=True)
        mask = self.lk.template.mask.mask if masked else None
        return MultipleImageViewer(figure_id, new_figure, self.image.n_dims,
                                   pixels_list, channels=channels,
                                   mask=mask).render(**kwargs)

    def view_appearances(self, figure_id=None, new_figure=False,
                         channels=None, masked=True, **kwargs):
        pixels_list = self.appearances(as_pixels=True)
        mask = self.lk.template.mask.mask if masked else None
        return MultipleImageViewer(figure_id, new_figure, self.image.n_dims,
                                   pixels_list, channels=channels,
                                   mask=mask).render(**kwargs)

    def view_error_images(self, figure_id=None, new_figure=False,
                          channels=None, masked=None, **kwargs):
        warped_images = self.warped_images(as_pixels=True)
        appearances = self.appearances(as_pixels=True)
        pixels_list = [a - i for a, i in zip(appearances, warped_images)]
        mask = self.lk.template.mask.mask if masked else None
        return MultipleImageViewer(figure_id, new_figure, self.image.n_dims,
                                   pixels_list, channels=channels,
                                   mask=mask).render(**kwargs)

    def view_final_fitting(self, figure_id=None, new_figure=False, **kwargs):
        image = deepcopy(self.image)
        image.landmarks['fitting'] = self.final_target
        return image.landmarks['fitting'].view(
            figure_id=figure_id, new_figure=new_figure).render(**kwargs)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        pixels_to_view = self.image.pixels
        targets_to_view = self.targets(as_points=True)
        return FittingViewer(figure_id, new_figure, self.image.n_dims,
                             pixels_to_view, targets_to_view).render(**kwargs)

