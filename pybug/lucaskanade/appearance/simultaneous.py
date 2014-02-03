from scipy.linalg import norm
import numpy as np
from pybug.lucaskanade.appearance.base import AppearanceLucasKanade


class SimultaneousForwardAdditive(AppearanceLucasKanade):

    @property
    def type(self):
        return 'SimFA'

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        n_iters = 0

        # Number of shape parameters
        n_params = self.transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = self.appearance_model._jacobian.T

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform,
                                       self._interpolator))

            # Concatenate VI_dW_dp with appearance model Jacobian
            self._J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            parameters = self.transform.as_vector() + delta_p[:n_params]
            self.transform.from_vector_inplace(parameters)
            lk_fitting.parameters.append(parameters)

            # Update appearance weights
            weights -= delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.status = 'completed'
        return self.transform


class SimultaneousForwardCompositional(AppearanceLucasKanade):

    @property
    def type(self):
        return 'SimFC'

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.transform.jacobian(
            self.template.mask.true_indices)

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        n_iters = 0

        # Number of shape parameters
        n_params = self.transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = self.appearance_model._jacobian.T

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Concatenate VI_dW_dp with appearance model Jacobian
            self._J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(delta_p[:n_params])
            lk_fitting.parameters.append(self.transform.as_vector())

            # Update appearance weights
            weights -= delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.status = 'completed'
        return self.transform


class SimultaneousInverseCompositional(AppearanceLucasKanade):

    @property
    def type(self):
        return 'SimIC'

    def _precompute(self):
        # Compute the Jacobian of the warp
        self._dW_dp = self.transform.jacobian(
            self.appearance_model.mean.mask.true_indices)

    def _align(self, lk_fitting, max_iters=20, project=True):
        # Initial error > eps
        error = self.eps + 1
        image = lk_fitting.image
        n_iters = 0

        # Number of shape parameters
        n_params = self.transform.n_parameters

        # Initial appearance weights
        if project:
            # Obtained weights by projection
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)
            weights = self.appearance_model.project(IWxp)
            # Reset template
            self.template = self.appearance_model.instance(weights)
        else:
            # Set all weights to 0 (yielding the mean)
            weights = np.zeros(self.appearance_model.n_active_components)

        # Compute appearance model Jacobian wrt weights
        appearance_jacobian = -self.appearance_model._jacobian.T

        # Baker-Matthews, Inverse Compositional Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current parameters
            IWxp = image.warp_to(self.template.mask, self.transform,
                                 interpolator=self._interpolator)

            # Compute steepest descent images, VT_dW_dp
            J = self.residual.steepest_descent_images(self.template,
                                                      self._dW_dp)

            # Concatenate VI_dW_dp with appearance model Jacobian
            self._J = np.hstack((J, appearance_jacobian))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = -np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.transform.compose_after_from_vector_inplace(delta_p[:n_params])
            lk_fitting.parameters.append(self.transform.as_vector())

            # Update appearance weights
            weights -= delta_p[n_params:]
            self.template = self.appearance_model.instance(weights)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        lk_fitting.status = 'completed'
        return self.transform
