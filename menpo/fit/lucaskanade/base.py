from __future__ import division
import abc
import numpy as np

from menpo.fit.base import Fitter
from menpo.fit.fittingresult import ParametricFittingResult


class LucasKanade(Fitter):
    r"""
    An abstract base class for implementations of Lucas-Kanade [1]_
    type algorithms.

    This is to abstract away optimisation specific functionality such as the
    calculation of the Hessian (which could be derived using a number of
    techniques, currently only Gauss-Newton).

    Parameters
    ----------
    image : :map:`Image`
        The image to perform the alignment upon.

        .. note:: Only the image is expected within the base class because
            different algorithms expect different kinds of template
            (image/model)
    residual : :map:`Residual`
    residual : :map:`Residual`
        The kind of residual to be calculated. This is used to quantify the
        error between the input image and the reference object.
    transform : :map:`Alignment`
        The transformation type used to warp the image in to the appropriate
        reference frame. This is used by the warping function to calculate
        sub-pixel coordinates of the input image in the reference frame.
    warp : function
        A function that takes 3 arguments,

            warp(:map:`Image`, :map:`Image`, :map:`Alignment`)

        This function is intended to perform sub-pixel interpolation of the
        pixel locations calculated by transforming the given image into the
        reference frame of the template. Appropriate functions are given in
        :module:`menpo.image.interpolation`.
    optimisation : ('GN',), optional
        A tuple containing the optimisation technique used to calculate the
        Hessian approximation. The first element is the string identifier and
        the remaining elements are any parameters for the technique.

        Default: ('GN',)
    eps : float, optional
        The convergence value. When calculating the level of convergence, if
        the norm of the delta parameter updates is less than `eps`, the
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
    ===== ==================== ===============================================

    Attributes
    ----------
    transform
    weights
    n_iters

    References
    ----------
    .. [1] Lucas, Bruce D., and Takeo Kanade.
       "An iterative image registration technique with an application to
       stereo vision." IJCAI. Vol. 81. 1981.
    """
    def __init__(self, residual, transform,
                 interpolator='scipy', optimisation=('GN',), eps=10**-10):
        # set basic state for all Lucas Kanade algorithms
        self.transform = transform
        self.residual = residual
        self.eps = eps
        self.interpolator = interpolator
        # select the optimisation approach and warp function
        self._calculate_delta_p = self._select_optimisation(optimisation)

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
        return np.linalg.solve(self._H, sd_delta_p)

    def _gauss_newton_lp_update(self, sd_delta_p):
        raise NotImplementedError("Gauss-Newton lp-norm optimization not "
                                  "implemented yet")

    def _levenberg_marquardt_update(self, sd_delta_p):
        raise NotImplementedError("Levenberg Marquardt optimization not "
                                  "implemented yet")

    def _set_up(self, **kwargs):
        pass

    def _create_fitting_result(self, image, parameters, gt_shape=None):
        return ParametricFittingResult(image, self, parameters=[parameters],
                                       gt_shape=gt_shape)

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        self.transform.from_vector_inplace(initial_parameters)
        return Fitter.fit(self, image, initial_parameters, gt_shape=gt_shape,
                          **kwargs)

    def get_parameters(self, shape):
        self.transform.set_target(shape)
        return self.transform.as_vector()
