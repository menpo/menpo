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
        self._initial_transform = transform
        self.TransformClass = transform.__class__
        self.image = image
        self.residual = residual
        self.update_step = update_step
        self.eps = eps
        self.transforms = [self._initial_transform]

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

    def align(self, max_iters=30):
        """
        Perform an alignment using the Lukas Kanade framework.
        :param max_iters: The maximum number of iterations that will be used
         in performing the alignment
        :return: The final transform that optimally aligns the source to the
         target.
        """
        self.transforms = [self._initial_transform]
        return self._align(max_iters)

    @abc.abstractmethod
    def _align(self, **kwargs):
        """
        The actual alignment function.
        """
        pass

    @property
    def optimal_transform(self):
        """
        The last transform that was applied is by definition the optimal
        """
        return self.transforms[-1]

    @property
    def transform_parameters(self):
        return [x.parameters for x in self.transforms]

    @property
    def n_iters(self):
        # nb at 0'th iteration we still have one transform
        # (self._initial_transform)
        return len(self.transforms) - 1


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

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Compute the Jacobian of the warp
        dW_dp = self.optimal_transform.jacobian(self.template.shape)

        # Compute steepest descent images, VT_dW_dp
        VT_dW_dp = self.residual.steepest_descent_images(self.template,
                                                             dW_dp)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(VT_dW_dp)

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self.template.shape,
                        self.optimal_transform,
                        interpolator=self._interpolator)

            # Compute steepest descent parameter updates
            sd_delta_p = (self.residual.
                          steepest_descent_update(VT_dW_dp, IWxp,
                                                  self.template))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.TransformClass.from_parameters(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform.inverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class AppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Compute the Jacobian of the warp (4)
        dW_dp = self.optimal_transform.jacobian(self._model.shape)

        # Compute steepest descent images, VT_dW_dp (3)
        VT_dW_dp = self.residual.steepest_descent_images(self._model, dW_dp)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(VT_dW_dp)

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self._model.shape,
                        self.optimal_transform,
                        interpolator=self._interpolator)

            # Compute steepest descent parameter updates
            sd_delta_p = (self.residual.
                          steepest_descent_update(VT_dW_dp, IWxp,
                                                  self._model))

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.TransformClass.from_parameters(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform.inverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ImageForwardAdditive(ImageLucasKanade):

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = warp(self.image, self.template.shape, self.optimal_transform,
                        interpolator=self._interpolator)

            # Compute the Jacobian of the warp
            dW_dp = self.optimal_transform.jacobian(self.template.shape)

            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = self.residual.steepest_descent_images(
                self.image, dW_dp, transform=self.optimal_transform,
                interpolator=self._interpolator)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                VI_dW_dp, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self.optimal_transform.parameters + delta_p
            self.transforms.append(
                self.TransformClass.from_parameters(new_params))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform