import abc
from scipy.linalg import norm, solve
import numpy as np
from pybug.warp import warp
from pybug.warp.base import map_coordinates_interpolator


class LucasKanade:
    __metaclass__ = abc.ABCMeta

    def __init__(self, image, residual, transform,
                 interpolator=map_coordinates_interpolator,
                 optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        # set basic state for all Lucas Kanade algorithms
        self.image = image
        self.transform = transform
        self.residual = residual
        self.update_step = update_step
        self.eps = eps
        self.warp_parameters = []
        self.n_iters = 0

        # select the optimisation approach and interpolator
        self._calculate_delta_p = self._select_optimisation(optimisation)
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
        H_lm = self._H + (self.update_step * LM)

        if self.residual.error < self.__e_lm:
            # Bad step, increase step
            self.update_step *= 10
        else:
            # Good step, decrease step
            self.update_step /= 10
            self.__e_lm = self.residual.error

        return solve(H_lm, sd_delta_p)

    @abc.abstractmethod
    def align(self):
        pass


class ImageLucasKanade(LucasKanade):

    def __init__(self, image, template, residual, transform,
                 interpolator=map_coordinates_interpolator,
                 optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        super(ImageLucasKanade, self).__init__(
            image, residual, transform, interpolator,
            optimisation, update_step, eps)
        # in image alignment, we align a template image to the target image
        self.template = template


class AppearanceModelLucasKanade(LucasKanade):

    def __init__(self, image, model, residual, transform,
                 interpolator=map_coordinates_interpolator,
                 optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        super(AppearanceModelLucasKanade, self).__init__(
            image, residual, transform, interpolator,
            optimisation, update_step, eps)
        # in appearance alignment, we align an appearance model to the target
        # image
        self._model = model


class ImageInverseCompositional(ImageLucasKanade):

    def align(self, max_iters=30):
        self.warp_parameters = []
        self.n_iters = 0
        # Initial error > eps
        error = self.eps + 1

        # Compute the Jacobian of the warp
        dW_dp = self.transform.jacobian(self.template.shape)

        # Compute steepest descent images, VT_dW_dp
        VT_dW_dp = self.residual.steepest_descent_images(self.template,
                                                             dW_dp)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(VT_dW_dp)

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self.template.shape, self.transform,
                        interpolator=self._interpolator)

            self.warp_parameters.append(self.transform.parameters)

            # Compute steepest descent parameter updates
            sd_delta_p = (self.residual.
                          steepest_descent_update(VT_dW_dp, IWxp,
                                                  self.template))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            deltap_transform = self.transform.from_parameters(delta_p)
            self.transform = self.transform.compose(deltap_transform.inverse)

            # Increase iteration count
            self.n_iters += 1

            # Test convergence
            error = np.abs(norm(delta_p))

        # Append final warp params
        self.warp_parameters.append(self.transform.parameters)

        return self.transform


class AppearanceInverseCompositional(AppearanceModelLucasKanade):

    def align(self, max_iters=30):
        self.warp_parameters = []
        self.n_iters = 0
        # Initial error > eps
        error = self.eps + 1

        # Compute the Jacobian of the warp (4)
        dW_dp = self.transform.jacobian(self._model.shape)

        # Compute steepest descent images, VT_dW_dp (3)
        VT_dW_dp = self.residual.steepest_descent_images(self._model, dW_dp)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(VT_dW_dp)

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self._template.shape, self.transform,
                        interpolator=self._interpolator)

            self.warp_parameters.append(self.transform.parameters)

            # Compute steepest descent parameter updates
            sd_delta_p = (self.residual.
                          steepest_descent_update(VT_dW_dp, IWxp,
                                                  self._model))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            deltap_transform = self.transform.from_parameters(delta_p)
            self.transform = self.transform.compose(deltap_transform.inverse)

            # Increase iteration count
            self.n_iters += 1

            # Test convergence
            error = np.abs(norm(delta_p))

        # Append final warp params
        self.warp_parameters.append(self.transform.parameters)

        return self.transform


class ImageForwardAdditive(ImageLucasKanade):

    def align(self, max_iters=30):
        self.warp_parameters = []
        self.n_iters = 0
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self.template.shape, self.transform,
                        interpolator=self._interpolator)

            self.warp_parameters.append(self.transform.parameters)

            # Compute the Jacobian of the warp
            dW_dp = self.transform.jacobian(self.template.shape)

            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = self.residual.steepest_descent_images(
                self.image, dW_dp, transform=self.transform,
                interpolator=self._interpolator)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                VI_dW_dp, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self.transform.parameters + delta_p
            self.transform = self.transform.from_parameters(new_params)

            # Increase iteration count
            self.n_iters += 1

            # Test convergence
            error = np.abs(norm(delta_p))

         # Append final warp params
        self.warp_parameters.append(self.transform.parameters)

        return self.transform