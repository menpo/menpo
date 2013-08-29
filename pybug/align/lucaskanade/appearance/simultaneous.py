from scipy.linalg import norm
import numpy as np
from pybug.align.lucaskanade.appearance.base import AppearanceLucasKanade


class SimultaneousForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=30, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Number of shape parameters
        n_params = self.optimal_transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = self.appearance_model._jacobian.T

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Compute warp Jacobian
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._warp))

            # Concatenate VI_dW_dp with appearance model Jacobian
            J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = -np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            params = self.optimal_transform.as_vector() + delta_p[:n_params]
            self.transforms.append(
                self.initial_transform.from_vector(params))

            # Update appearance weights
            weights -= delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class SimultaneousForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=30, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Number of shape parameters
        n_params = self.optimal_transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = self.appearance_model._jacobian.T

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Concatenate VI_dW_dp with appearance model Jacobian
            J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = -np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(
                delta_p[:n_params])
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform))

            # Update appearance weights
            weights -= delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class SimultaneousInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute the Jacobian of the warp
        self._dW_dp = self.initial_transform.jacobian(
            self.appearance_model.mean.mask.true_indices)

        pass

    def _align(self, max_iters=30, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Number of shape parameters
        n_params = self.optimal_transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = self.appearance_model._jacobian.T

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Compute steepest descent images, VT_dW_dp
            J = self.residual.steepest_descent_images(self.template,
                                                      self._dW_dp)

            # Concatenate VI_dW_dp with appearance model Jacobian
            J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(
                delta_p[:n_params])
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform.inverse))

            # Update appearance weights
            weights = weights + delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform
