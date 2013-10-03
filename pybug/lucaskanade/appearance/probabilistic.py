from scipy.linalg import norm
import numpy as np
from pybug.lucaskanade.appearance.base import AppearanceLucasKanade


class ProbabilisticForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._interpolator))

            # Project out appearance model from VT_dW_dp
            self._J = (self.appearance_model._to_subspace(J.T) +
                       self.appearance_model._within_subspace(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            params = self.optimal_transform.as_vector() + delta_p
            self.initial_transform.update_from_vector(params)
            self.transforms.append(self.initial_transform)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ProbabilisticForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = (self.appearance_model._to_subspace(J.T) +
                       self.appearance_model._within_subspace(J.T)).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(self.initial_transform))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ProbabilisticInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        J = self.residual.steepest_descent_images(self.template,
                                                  self._dW_dp)
        # Project out appearance model from VT_dW_dp
        self._J = (self.appearance_model._to_subspace(J.T) +
                   self.appearance_model._within_subspace(J.T)).T
        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(self._J, J2=J)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(
                    self.initial_transform.pseudoinverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ToSubspaceForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._interpolator))

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._to_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            params = self.optimal_transform.as_vector() + delta_p
            self.initial_transform.update_from_vector(params)
            self.transforms.append(self.initial_transform)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ToSubspaceForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._to_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(self.initial_transform))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ToSubspaceInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VT_dW_dp
            J = self.residual.steepest_descent_images(self.template,
                                                          self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._to_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(
                    self.initial_transform.pseudoinverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class WithinSubspaceForwardAdditive(AppearanceLucasKanade):

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute warp Jacobian
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._interpolator))

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._within_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            params = self.optimal_transform.as_vector() + delta_p
            self.initial_transform.update_from_vector(params)
            self.transforms.append(self.initial_transform)

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class WithinSubspaceForwardCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VI_dW_dp
            J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._within_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose( self.initial_transform))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class WithinSubspaceInverseCompositional(AppearanceLucasKanade):

    def _precompute(self):
        # Compute warp Jacobian
        self._dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        pass

    def _align(self, max_iters=50, project=True):
        # Initial error > eps
        error = self.eps + 1

        # Baker-Matthews, Inverse Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self.image.warp_to(self.template.mask,
                                      self.optimal_transform,
                                      interpolator=self._interpolator)

            # Compute steepest descent images, VT_dW_dp
            J = self.residual.steepest_descent_images(self.template,
                                                      self._dW_dp)

            # Project out appearance model from VT_dW_dp
            self._J = self.appearance_model._within_subspace(J.T).T

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J, J2=J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            self.initial_transform.update_from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(
                    self.initial_transform.pseudoinverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform
