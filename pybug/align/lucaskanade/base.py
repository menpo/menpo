import abc
from scipy.linalg import norm, solve
import numpy as np
from pybug.warp import warp
from pybug.warp.base import map_coordinates_interpolator


class LucasKanade:
    __metaclass__ = abc.ABCMeta

    def __init__(self, image, template, residual, transform,
                 interpolator=map_coordinates_interpolator,
                 optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        self._image = image
        self._template = template
        self._transform = transform
        self._residual = residual
        self._calculate_delta_p = self._select_optimisation(optimisation)
        self._update_step = update_step
        self._eps = eps
        self._warp_parameters = []
        self._interpolator = interpolator

    def _select_optimisation(self, optimisation):
        if optimisation is 'GN':
            return self._gauss_newton_update
        elif optimisation is 'LM':
            self.__e_lm = 0
            return self._levenberg_marquardt_update
        else:
            raise ValueError('Unknown optimisation string selected. Valid'
                             'options are: GN, LM')

    def _gauss_newton_update(self, sd_delta_p):
        return solve(self._H, sd_delta_p)

    def _levenberg_marquardt_update(self, sd_delta_p):
        LM = np.diagflat(np.diagonal(self._H))
        H_lm = self._H + (self._update_step * LM)

        if self._residual.error < self.__e_lm:
            # Bad step, increase step
            self._update_step *= 10
        else:
            # Good step, decrease step
            self._update_step /= 10
            self.__e_lm = self._residual.error

        return solve(H_lm, sd_delta_p)

    @abc.abstractmethod
    def align(self):
        pass

    @property
    def warp_parameters(self):
        return self._warp_parameters

    @property
    def iteration_count(self):
        return self._iters


class InverseCompositional(LucasKanade):

    def align(self, max_iters=30):
        self._warp_parameters = []
        self._iters = 0
        # Initial error > eps
        error = self._eps + 1

        # Compute the Jacobian of the warp
        dW_dp = self._transform.jacobian(self._template.shape)

        # Compute steepest descent images, VT_dW_dp
        VT_dW_dp = self._residual.steepest_descent_images(self._template,
                                                             dW_dp)

        # Compute Hessian and inverse
        self._H = self._residual.calculate_hessian(VT_dW_dp)

        # Baker-Matthews, Inverse Compositional Algorithm
        while self._iters < (max_iters - 1) and error > self._eps:
            # Compute warped image with current parameters
            IWxp = warp(self._image, self._template.shape, self._transform,
                        interpolator=self._interpolator)

            self._warp_parameters.append(self._transform.parameters)

            # Compute steepest descent parameter updates
            sd_delta_p = (self._residual.
                          steepest_descent_update(VT_dW_dp, IWxp,
                                                  self._template))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            deltap_transform = self._transform.from_parameters(delta_p)
            self._transform = self._transform.compose(deltap_transform.inverse)

            # Increase iteration count
            self._iters += 1

            # Test convergence
            error = np.abs(norm(delta_p))

        # Append final warp params
        self._warp_parameters.append(self._transform.parameters)

        return self._transform


class ForwardAdditive(LucasKanade):

    def align(self, max_iters=30):
        self._warp_parameters = []
        self._iters = 0
        # Initial error > eps
        error = self._eps + 1

        # Forward Additive Algorithm
        while self._iters < (max_iters - 1) and error > self._eps:
            # Compute warped image with current parameters
            IWxp = warp(self._image, self._template.shape, self._transform,
                        interpolator=self._interpolator)

            self._warp_parameters.append(self._transform.parameters)

            # Compute the Jacobian of the warp
            dW_dp = self._transform.jacobian(self._template.shape)

            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = (self._residual.
                        steepest_descent_images(self._image, dW_dp,
                            transform=self._transform,
                            interpolator=self._interpolator))

            # Compute Hessian and inverse
            self._H = self._residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = (self._residual.
                          steepest_descent_update(VI_dW_dp, self._template,
                                                  IWxp))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self._transform.parameters + delta_p
            self._transform = self._transform.from_parameters(new_params)

            # Increase iteration count
            self._iters += 1

            # Test convergence
            error = np.abs(norm(delta_p))

         # Append final warp params
        self._warp_parameters.append(self._transform.parameters)

        return self._transform