import abc
from scipy.linalg import norm, solve
import numpy as np
from pybug.warp.base import scipy_warp


class LucasKanade(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, residual, transform,
                 warp=scipy_warp, optimisation='GN', update_step=0.001,
                 eps=10 ** -10):
        # set basic state for all Lucas Kanade algorithms
        self.initial_transform = transform
        self.TransformClass = transform.__class__
        self.residual = residual
        self.update_step = update_step
        self.eps = eps

        # select the optimisation approach and warp function
        self._calculate_delta_p = self._select_optimisation(optimisation)
        self._warp = warp

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

    def _precompute(self):
        """
        Performs pre-computations related to specific alignment algorithms
        """
        pass

    def align(self, image, params, max_iters=30):
        """
        Perform an alignment using the Lukas Kanade framework.
        :param max_iters: The maximum number of iterations that will be used
         in performing the alignment
        :return: The final transform that optimally aligns the source to the
         target.
        """
        # TODO: define a consistent multi-resolution logic
        self.transforms = [self.initial_transform.from_vector(params)]
        self.image = image
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
        return [x.as_vector() for x in self.transforms]

    @property
    def n_iters(self):
        # nb at 0'th iteration we still have one transform
        # (self.initial_transform)
        return len(self.transforms) - 1


class ImageLucasKanade(LucasKanade):

    def __init__(self, template, residual, transform,
                 warp=scipy_warp, optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        super(ImageLucasKanade, self).__init__(
            residual, transform,
            warp, optimisation, update_step, eps)
        # in image alignment, we align a template image to the target image
        self.template = template

        # pre-compute
        self._precompute()


class AppearanceModelLucasKanade(LucasKanade):

    def __init__(self, model, residual, transform,
                 warp=scipy_warp, optimisation='GN', update_step=0.001,
                 eps=10 ** -6):
        super(AppearanceModelLucasKanade, self).__init__(
            residual, transform,
            warp, optimisation, update_step, eps)
        # in appearance model alignment, we align an appearance model to the
        # target image
        self.appearance_model = model

        # pre-compute
        self._precompute()


class ImageForwardAdditive(ImageLucasKanade):

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Forward Additive Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # Compute the Jacobian of the warp
            dW_dp = self.optimal_transform.jacobian(
                self.template.mask.true_indices)

            # TODO: rename kwarg "forward" to "forward_additive"
            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(self.template,
                                            self.optimal_transform,
                                            self._warp))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                VI_dW_dp, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self.optimal_transform.as_vector() + delta_p
            self.transforms.append(
                self.initial_transform.from_vector(new_params))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ImageForwardCompositional(ImageLucasKanade):

    def _precompute(self):
        # Compute the Jacobian of the warp
        self.dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # Forward Compositional Algorithm
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, self.template,
                              self.optimal_transform)

            # TODO: add "forward_compositional" kwarg with options
            # In the forward compositional algorithm there are two different
            # ways of computing the steepest descent images:
            #   1. V[I(x)](W(x,p)) * dW/dx * dW/dp
            #   2. V[I(W(x,p))] * dW/dp -> this is what is currently used
            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = self.residual.steepest_descent_images(IWxp, self.dW_dp)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                VI_dW_dp, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ImageInverseCompositional(ImageLucasKanade):

    def _precompute(self):
        # Compute the Jacobian of the warp
        dW_dp = self.initial_transform.jacobian(
            self.template.mask.true_indices)

        # Compute steepest descent images, VT_dW_dp
        self.VT_dW_dp = self.residual.steepest_descent_images(
            self.template, dW_dp)

        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(self.VT_dW_dp)

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
                self.VT_dW_dp, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            delta_p_transform = self.initial_transform.from_vector(delta_p)
            self.transforms.append(
                self.optimal_transform.compose(delta_p_transform.inverse))

            # Test convergence
            error = np.abs(norm(delta_p))

        return self.optimal_transform


class ProjectOutAppearanceForwardAdditive(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        # Initial error > eps
        error = self.eps + 1

        # grab mean appearance pixel locations
        # project out the mean appearance immediately
        # TODO implement project_out and uncomment this
        # mean_appearance = self.appearance_model.project_out(
        #     self.appearance_model.mean)
        mean_appearance = self.appearance_model.mean

        # Forward Additive Algorithm
        ims =[]  # tmp
        while self.n_iters < (max_iters - 1) and error > self.eps:
            # Compute warped image with current parameters
            IWxp = self._warp(self.image, mean_appearance,
                              self.optimal_transform)
            ims.append(IWxp)  # tmp
            # and project out the appearance model from this image
            # TODO implement project_out and uncomment this
            # IWxp = self.appearance_model.project_out(IWxp)
            # Compute the Jacobian of the warp
            dW_dp = self.optimal_transform.jacobian(
                mean_appearance.mask.true_indices)

            # Compute steepest descent images, VI_dW_dp
            VI_dW_dp = self.residual.steepest_descent_images(
                self.image, dW_dp, forward=(mean_appearance,
                                            self.optimal_transform,
                                            self._warp))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(VI_dW_dp)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                VI_dW_dp, mean_appearance, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp parameters
            new_params = self.optimal_transform.as_vector() + delta_p
            self.transforms.append(
                self.initial_transform.from_vector(new_params))

            # Test convergence
            error = np.abs(norm(delta_p))
            print '{} - convergence: {}'.format(self.n_iters, error)
            print '{} - residualerr: {}\n'.format(self.n_iters,
                                                  self.residual.error)

        #return self.optimal_transform
        return ims  # tmp


class ProjectOutAppearanceForwardCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class ProjectOutAppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class SimultaneousAppearanceForwardAdditive(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class SimultaneousAppearanceForwardCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class SimultaneousAppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AlternatingAppearanceForwardAdditive(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AlternatingAppearanceForwardCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AlternatingAppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AdaptiveAppearanceForwardAdditive(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AdaptiveAppearanceForwardCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class AdaptiveAppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class ProbabilisticAppearanceForwardAdditive(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class ProbabilisticAppearanceForwardCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass


class ProbabilisticAppearanceInverseCompositional(AppearanceModelLucasKanade):

    def _align(self, max_iters=30):
        pass