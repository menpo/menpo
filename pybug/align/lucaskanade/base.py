import abc
from scipy.linalg import norm, solve
import numpy as np
from pybug.warp.base import scipy_warp


class LucasKanade(object):
    r"""
    An abstract base class for implementations of Lucas-Kanade [1]_
    type algorithms.

    This is to abstract away optimisation specific functionality such as the
    calculation of the Hessian (which could be derived using a number of
    techniques, including Gauss-Newton and Levenberg-Marquardt).

    Parameters
    ----------
    image : :class:`pybug.image.base.Image`
        The image to perform the alignment upon.

        .. note:: Only the image is expected within the base class because
            different algorithms expect different kinds of template
            (image/model)
    residual : :class:`pybug.align.lucaskanade.residual.Residual`
        The kind of residual to be calculated. This is used to quantify the
        error between the input image and the reference object.
    transform : :class:`pybug.transform.base.Transform`
        The transformation type used to warp the image in to the appropriate
        reference frame. This is used by the warping function to calculate
        sub-pixel coordinates of the input image in the reference frame.
    warp : function
        A function that takes 3 arguments,
        ``warp(`` :class:`image <pybug.image.base.Image>`,
        :class:`template <pybug.image.base.Image>`,
        :class:`transform <pybug.transform.base.Transform>` ``)``
        This function is intended to perform sub-pixel interpolation of the
        pixel locations calculated by transforming the given image into the
        reference frame of the template. Appropriate functions are given in
        :doc:`pybug.warp`.
    optimisation : 'GN' | 'LM', optional
        The optimisation technique used to calculate the Hessian approximation.

        Default: 'GN'
    update_step : float, optional
        The update step used when performing a Levenberg-Marquardt
        optimisation.

        Default: 0.001
    eps : float, optional
        The convergence value. When calculating the level of convergence, if
        the norm of the delta parameter updates is less than ``eps``, the
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
    'LM'  Levenberg-Marquardt  :math:`\mathbf{J^T J + \lambda\, diag(J^T J)}`
    ===== ==================== ===============================================

    Attributes
    ----------
    optimal_transform
    transform_parameters
    n_iters

    References
    ----------
    .. [1] Lucas, Bruce D., and Takeo Kanade.
       "An iterative image registration technique with an application to
       stereo vision." IJCAI. Vol. 81. 1981.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, residual, transform,
                 warp=scipy_warp, optimisation='GN', update_step=0.001,
                 eps=1 ** -10):
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
        r"""
        Perform an alignment using the Lukas-Kanade framework.

        Parameters
        ----------
        max_iters : int
            The maximum number of iterations that will be used in performing
            the alignment

        Returns
        -------
        transform : :class:`pybug.transform.base.Transform`
            The final transform that optimally aligns the source to the
            target.
        """
        # TODO: define a consistent multi-resolution logic
        self.transforms = [self.initial_transform.from_vector(params)]
        self.image = image
        return self._align(max_iters)

    @abc.abstractmethod
    def _align(self, **kwargs):
        r"""
        Abstract method to be overridden by subclasses that implements the
        alignment algorithm.
        """
        pass

    @property
    def optimal_transform(self):
        r"""
        The final transform that was applied is by definition the optimal.

        :type: :class:`pybug.transform.base.Transform`
        """
        return self.transforms[-1]

    @property
    def transform_parameters(self):
        r"""
         The parameters of every transform calculated during alignment.

        :type: list of (P,) ndarrays

        The parameters are obtained by calling the ``as_vector()`` method on
        each transform.
        """
        return [x.as_vector() for x in self.transforms]

    @property
    def n_iters(self):
        r"""
        The number of iterations performed.

        :type: int
        """
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