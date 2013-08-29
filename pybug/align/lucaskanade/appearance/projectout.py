from scipy.linalg import norm
import numpy as np
from pybug.align.lucaskanade.appearance.base import AppearanceLucasKanade


class ProjectOutForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Project out appearance model from mean appearance
        self.template = self.appearance_model.project_out(self.template)

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Project out appearance model from warped image
            IWxp = self.appearance_model.project_out(IWxp)

            # Compute warp Jacobian
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._warp))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = -np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self.optimal_transform.as_vector() + delta_p
            self.transforms.append(
                self.initial_transform.from_vector(new_params))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ProjectOutForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Project out appearance model from mean appearance
        self.template = self.appearance_model.project_out(self.template)

        # Forward Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Project out appearance model from warped image
            IWxp = self.appearance_model.project_out(IWxp)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = -np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ProjectOutInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        J = self.residual.steepest_descent_images(
            self.template, dW_dp)

        # Project out appearance model from VT_dW_dp
        self._J = self.appearance_model._project_out(J)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(self._J)

        pass

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform.inverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform