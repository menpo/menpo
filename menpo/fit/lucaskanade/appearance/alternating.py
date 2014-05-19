from scipy.linalg import norm
import numpy as np

from .base import AppearanceLucasKanade


class AlternatingForwardAdditive(AppearanceLucasKanade):

    @property
    def algorithm(self):
        return 'Alternating-FA'

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = [[0]]
        n_iters = 0

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self.interpolator)

            # Compute appearance
            weights = self.appearance_model.project(IWxp)
            self.template = self.appearance_model.instance(weights)
            fitting_result.weights.append(weights)

            # Compute warp Jacobian
            dW_dp = self.transform.d_dp(self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            self._J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform,
                                       self.interpolator))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            parameters = self.transform.as_vector() + delta_p
            self.transform.from_vector_inplace(parameters)
            fitting_result.parameters.append(parameters)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result


class AlternatingForwardCompositional(AppearanceLucasKanade):

    @property
    def algorithm(self):
        return 'Alternating-FC'

    def _set_up(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.d_dp(self.template.mask.true_indices)

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = [[0]]
        n_iters = 0

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self.interpolator)

            # Compute template by projection
            weights = self.appearance_model.project(IWxp)
            self.template = self.appearance_model.instance(weights)
            fitting_result.weights.append(weights)

            # Compute steepest descent images, VI_dW_dp
            self._J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result


class AlternatingInverseCompositional(AppearanceLucasKanade):

    @property
    def algorithm(self):
        return 'Alternating-IC'

    def _set_up(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.d_dp(self.template.mask.true_indices)

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        fitting_result.weights = [[0]]
        n_iters = 0

        # Baker-Matthews, Inverse Compositional Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self.interpolator)

            # Compute appearance
            weights = self.appearance_model.project(IWxp)
            self.template = self.appearance_model.instance(weights)
            fitting_result.weights.append(weights)

            # Compute steepest descent images, VT_dW_dp
            self._J = self.residual.steepest_descent_images(self.template,
                                                            self._dW_dp)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Request the pesudoinverse vector from the transform
            inv_delta_p = self.transform.pseudoinverse_vector(delta_p)

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(inv_delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result
