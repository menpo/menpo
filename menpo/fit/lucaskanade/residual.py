"""
This module contains a set of similarity measures that was designed for use
within the Lucas-Kanade framework. They therefore expose a number of methods
that make them useful for inverse compositional and forward additive
Lucas-Kanade.

These similarity measures are designed to be dimension independent where
possible. For this reason, some methods look more complicated than would be
normally the case. For example, calculating the Hessian involves summing
a multi-dimensional array, so we dynamically calculate the list of axes
to sum over. However, the basics of the logic, other than dimension
reduction, should be similar to the original algorithms.

References
----------

.. [1] Lucas, Bruce D., and Takeo Kanade.
       "An iterative image registration technique with an application to stereo
       vision."
       IJCAI. Vol. 81. 1981.
"""
import abc
import copy
import numpy as np
from numpy.fft import fftshift, fftn
import scipy.linalg

from menpo.math import log_gabor
from menpo.image import MaskedImage


class Residual(object):
    """
    An abstract base class for calculating the residual between two images
    within the Lucas-Kanade algorithm. The classes were designed
    specifically to work within the Lucas-Kanade framework and so no
    guarantee is made that calling methods on these subclasses will generate
    correct results.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def error(self):
        r"""
        The RMS of the error image.

        :type: float

        Notes
        -----
        Will only generate a result if the
        :func:`steepest_descent_update` function has been calculated prior.

        .. math::
            error = \sqrt{\sum_x E(x)^2}

        where :math:`E(x) = T(x) - I(W(x;p))` within the forward additive
        framework.
        """
        return np.sqrt(np.mean(self._error_img ** 2))

    @abc.abstractmethod
    def steepest_descent_images(self, image, dW_dp, **kwargs):
        r"""
        Calculates the standard steepest descent images.

        Within the forward additive framework this is defined as

        .. math::
             \nabla I \frac{\partial W}{\partial p}

        The input image is vectorised (`N`-pixels) so that masked images can
        be handled.

        Parameters
        ----------
        image : :class:`menpo.image.base.Image`
            The image to calculate the steepest descent images from, could be
            either the template or input image depending on which framework is
            used.
        dW_dp : ndarray
            The Jacobian of the warp.

        Returns
        -------
        VT_dW_dp : (N, n_params) ndarray
            The steepest descent images
        """
        pass

    @abc.abstractmethod
    def calculate_hessian(self, VT_dW_dp):
        r"""
        Calculates the Gauss-Newton approximation to the Hessian.

        This is abstracted because some residuals expect the Hessian to be
        pre-processed. The Gauss-Newton approximation to the Hessian is
        defined as:

        .. math::
            \mathbf{J J^T}

        Parameters
        ----------
        VT_dW_dp : (N, n_params) ndarray
            The steepest descent images.

        Returns
        -------
        H : (n_params, n_params) ndarray
            The approximation to the Hessian
        """
        pass

    @abc.abstractmethod
    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        r"""
        Calculates the steepest descent parameter updates.

        These are defined, for the forward additive algorithm, as:

        .. math::
            \sum_x [ \nabla I \frac{\partial W}{\partial p} ]^T [ T(x) - I(W(x;p)) ]

        Parameters
        ----------
        VT_dW_dp : (N, n_params) ndarray
            The steepest descent images.
        IWxp : :class:`menpo.image.base.Image`
            Either the warped image or the template
            (depending on the framework)
        template : :class:`menpo.image.base.Image`
            Either the warped image or the template
            (depending on the framework)

        Returns
        -------
        sd_delta_p : (n_params,) ndarray
            The steepest descent parameter updates.
        """
        pass

    def _calculate_gradients(self, image, forward=None):
        r"""
        Calculates the gradients of the given method.

        If `forward` is provided, then the gradients are warped
        (as required in the forward additive algorithm)

        Parameters
        ----------
        image : :class:`menpo.image.base.Image`
            The image to calculate the gradients for
        forward : (:class:`template <menpo.image.base.Image>`, :class:`template <menpo.transform.base.AlignableTransform>`, `warp`), optional
            A tuple containing the extra weights required for the function
            `warp` (which should be passed as a function handle).

            Default: `None`
        """
        if forward:
            # Calculate the gradient over the image
            gradient = image.gradient()
            # Warp gradient for forward additive, if we've been given a
            # transform
            template, transform, interpolator = forward
            gradient = gradient.warp_to(template.mask, transform,
                                        interpolator=interpolator)
        else:
            # Calculate the gradient over the image and remove one pixels at
            # the borders of the image mask
            gradient = image.gradient(nullify_values_at_mask_boundaries=True)

        return gradient


class LSIntensity(Residual):

    type = 'SSD'

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # compute gradient
        # gradient:  height  x  width  x  (n_channels x n_dims)
        gradient = self._calculate_gradients(image, forward=forward)

        # reshape gradient
        # gradient:  n_pixels  x  (n_channels x n_dims)
        gradient = gradient.as_vector(keep_channels=True)

        # reshape gradient
        # gradient:  n_pixels  x  n_channels  x  n_dims
        gradient = np.reshape(gradient, (-1, image.n_channels,
                                         image.n_dims))

        # compute steepest descent images
        # gradient:  n_pixels  x  n_channels  x            x  n_dims
        # dW_dp:     n_pixels  x              x  n_params  x  n_dims
        # sdi:       n_pixels  x  n_channels  x  n_params
        sdi = np.sum(dW_dp[:, None, :, :] * gradient[:, :, None, :], axis=3)

        # reshape steepest descent images
        # sdi:  (n_pixels x n_channels)  x  n_params
        return np.reshape(sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, J, J2=None):
        if J2 is None:
            H = J.T.dot(J)
        else:
            H = J.T.dot(J2)
        return H

    def steepest_descent_update(self, sdi, IWxp, template):
        self._error_img = IWxp.as_vector() - template.as_vector()
        return sdi.T.dot(self._error_img)


class GaborFourier(Residual):

    type = 'GaborFourier'

    def __init__(self, image_shape, **kwargs):
        if 'filter_bank' in kwargs:
            self._filter_bank = kwargs.get('filter_bank')
            if self._filter_bank.shape != image_shape:
                raise ValueError('Filter bank shape must match the shape '
                                 'of the image')
        else:
            gabor = log_gabor(np.ones(image_shape), **kwargs)
            # Get filter bank matrix
            self._filter_bank = gabor[2]

        # Flatten the filter bank for vectorized calculations
        self._filter_bank = self._filter_bank.flatten()

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # compute gradient
        # gradient:  height  x  width  x  n_channels
        gradient_img = self._calculate_gradients(image, forward=forward)

        # reshape gradient
        # gradient:  n_pixels  x  (n_channels x n_dims)
        gradient = gradient_img.as_vector(keep_channels=True)

        # reshape gradient
        # gradient:  n_pixels  x  n_channels  x  n_dims
        gradient = np.reshape(gradient, (-1, image.n_channels, image.n_dims))

        # compute steepest descent images
        # gradient:  n_pixels  x  n_channels  x            x  n_dims
        # dW_dp:     n_pixels  x              x  n_params  x  n_dims
        # sdi:       n_pixels  x  n_channels  x  n_params
        sdi = np.sum(dW_dp[:, None, :, :] * gradient[:, :, None, :], axis=3)

        # make sdi images
        # sdi_img:  shape  x  n_channels  x  n_params
        sdi_img_channels = image.n_channels * dW_dp.shape[1]
        sdi_img = MaskedImage.blank(gradient_img.shape,
                                      n_channels=sdi_img_channels,
                                      mask=gradient_img.mask)
        sdi_img.from_vector_inplace(sdi.flatten())

        # compute FFT over each channel, parameter and dimension
        # fft_sdi:  height  x  width  x  n_channels  x  n_params
        fft_axes = range(image.n_dims)
        fft_sdi = fftshift(fftn(sdi_img.pixels, axes=fft_axes), axes=fft_axes)

        # ToDo: Note that, fft_sdi is rectangular, i.e. is not define in
        # terms of the mask pixels, but in terms of the whole image.
        # Selecting mask pixels once the fft has been computed makes no
        # sense because they have lost their original spatial meaning.

        # reshape steepest descent images
        # sdi:  (height x width x n_channels)  x  n_params
        return np.reshape(fft_sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, sdi):
        # reshape steepest descent images
        # sdi:  n_channels  x  n_pixels  x  n_params
        sdi = np.reshape(sdi, (-1, self._filter_bank.shape[0], sdi.shape[1]))

        # compute filtered steepest descent images
        # _filter_bank:              x  n_pixels  x
        # sdi:           n_channels  x  n_pixels  x  n_params
        # filtered_sdi:  n_channels  x  n_pixels  x  n_params
        filtered_sdi = (self._filter_bank[None, ..., None] ** 0.5) * sdi

        # reshape filtered steepest descent images
        # filtered_sdi:  (n_pixels x n_channels)  x  n_params
        filtered_sdi = np.reshape(filtered_sdi, (-1, sdi.shape[-1]))

        # compute filtered hessian
        # filtered_sdi:  (n_pixels x n_channels)  x  n_params
        # hessian:              n_param           x  n_param
        return np.conjugate(filtered_sdi).T.dot(filtered_sdi)

    def steepest_descent_update(self, sdi, IWxp, template):
        # compute error image
        # error_img:  height  x  width  x  n_channels
        error_img = IWxp.pixels - template.pixels

        # compute FFT error image
        # fft_error_img:  height  x  width  x  n_channels
        fft_axes = range(IWxp.n_dims)
        fft_error_img = fftshift(fftn(error_img, axes=fft_axes),
                                 axes=fft_axes)

        # reshape FFT error image
        # fft_error_img:  (height x width)  x  n_channels
        fft_error_img = np.reshape(fft_error_img, (-1, IWxp.n_channels))

        # compute filtered steepest descent images
        # _filter_bank:        (height x width)  x
        # fft_error_img:       (height x width)  x  n_channels
        # filtered_error_img:  (height x width)  x  n_channels
        filtered_error_img = (self._filter_bank[..., None] * fft_error_img)

        # reshape _error_img
        # _error_img:  (height x width x n_channels)
        self._error_img = filtered_error_img.flatten()

        # compute steepest descent update
        # sdi:         (height x width x n_channels)  x  n_parameters
        # _error_img:  (height x width x n_channels)
        # sdu:             n_parameters
        return sdi.T.dot(np.conjugate(self._error_img))


class ECC(Residual):

    type = 'ECC'

    def __normalise_images(self, image):
        # TODO: do we need to copy the image?
        # ToDo: is this supposed to be per channel normalization?
        new_im = copy.deepcopy(image)
        i = new_im.pixels
        i -= np.mean(i)
        i /= scipy.linalg.norm(i)
        i = np.nan_to_num(i)

        new_im.pixels = i
        return new_im

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # normalize image
        # image:  height  x  width  x  n_channels
        norm_image = self.__normalise_images(image)

        # compute gradient
        # gradient:  height  x  width  x  n_channels
        gradient = self._calculate_gradients(norm_image, forward=forward)

        # reshape gradient
        # gradient:  n_pixels  x  (n_channels x n_dims)
        gradient = gradient.as_vector(keep_channels=True)

        # reshape gradient
        # gradient:  n_pixels  x  n_channels  x  n_dims
        gradient = np.reshape(gradient, (-1, image.n_channels,
                                         image.n_dims))

        # compute steepest descent images
        # gradient:  n_pixels  x  n_channels  x            x  n_dims
        # dW_dp:     n_pixels  x              x  n_params  x  n_dims
        # sdi:       n_pixels  x  n_channels  x  n_params
        sdi = np.sum(dW_dp[:, None, :, :] * gradient[:, :, None, :], axis=3)

        # reshape steepest descent images
        # sdi:  (n_pixels x n_channels)  x  n_params
        return np.reshape(sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, sdi):
        H = sdi.T.dot(sdi)
        self._H_inv = scipy.linalg.inv(H)
        return H

    def steepest_descent_update(self, sdi, IWxp, template):
        normalised_IWxp = self.__normalise_images(IWxp).as_vector()
        normalised_template = self.__normalise_images(template).as_vector()

        Gt = sdi.T.dot(normalised_template)
        Gw = sdi.T.dot(normalised_IWxp)

        # Calculate the numerator
        IWxp_norm = scipy.linalg.norm(normalised_IWxp)
        num = (IWxp_norm ** 2) - np.dot(Gw.T, np.dot(self._H_inv, Gw))

        # Calculate the denominator
        den1 = np.dot(normalised_template, normalised_IWxp)
        den2 = np.dot(Gt.T, np.dot(self._H_inv, Gw))
        den = den1 - den2

        # Calculate lambda to choose the step size
        # Avoid division by zero
        if den > 0:
            l = num / den
        else:
            # TODO: Should be other step described in paper
            l = 0

        self._error_img = l * normalised_IWxp - normalised_template

        return sdi.T.dot(self._error_img)


class GradientImages(Residual):

    type = 'GradientImages'

    def __regularise_gradients(self, gradients):
        pixels = gradients.pixels
        ab = np.sqrt(np.sum(np.square(pixels), -1))
        m_ab = np.median(ab)
        ab = ab + m_ab
        gradients.pixels = pixels / ab[..., None]
        return gradients

    def steepest_descent_images(self, image, dW_dp, forward=None):
        n_dims = image.n_dims

        # compute gradient
        # gradient:  height  x  width  x  (n_channels x n_dims)
        first_grad = self._calculate_gradients(image, forward=forward)
        self.__template_gradients = self.__regularise_gradients(first_grad)

        # compute second order derivatives over each dimension
        # gradient:  height  x  width  x  (n_channels x n_dims x n_dims)
        second_grad = self._calculate_gradients(self.__template_gradients)

        # reshape gradient
        # second_grad:  n_pixels  x  (n_channels x n_dims)
        second_grad = second_grad.as_vector(keep_channels=True).copy()

        # reshape gradient
        # second_grad:  n_pixels  x  n_channels  x  n_dims  x  n_dims
        second_grad = np.reshape(second_grad, (-1, image.n_channels,
                                               n_dims, n_dims))

        # Fix crossed derivatives: dydx = dxdy
        second_grad[:, :, 1, 0] = second_grad[:, :, 0, 1]

        # compute steepest descent images
        # second_grad: n_pixels  x  n_channels  x            x n_dims x n_dims
        # dW_dp:       n_pixels  x              x  n_params  x n_dims x
        # sdi:         n_pixels  x  n_channels  x  n_params
        sdi = np.sum(dW_dp[:, None, :, :, None] *
                     second_grad[:, :, None, :, :], axis=3).swapaxes(2, 3)

        # reshape steepest descent images
        # sdi:  (n_pixels x n_channels)  x  n_params
        return np.reshape(sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, sdi):
        # compute hessian
        # sdi:      (n_pixels x n_channels x n_dims)  x  n_parameters
        # hessian:             n_parameters           x  n_parameters
        return sdi.T.dot(sdi)

    def steepest_descent_update(self, sdi, IWxp, template):
        # compute IWxp regularized gradient
        # gradient:  height  x  width  x  (n_channels x n_dims)
        IWxp_gradient = self._calculate_gradients(IWxp)
        IWxp_gradient = self.__regularise_gradients(IWxp_gradient)

        # compute vectorized error_image
        # _error_img:  (n_pixels x n_channels x n_dims)
        self._error_img = (IWxp_gradient.as_vector() -
                           self.__template_gradients.as_vector())

        # compute steepest descent update
        # sdi:         (n_pixels x n_channels x n_dims)  x  n_parameters
        # _error_img:  (n_pixels x n_channels x n_dims)
        # sdu:                    n_parameters
        return sdi.T.dot(self._error_img)


class GradientCorrelation(Residual):

    type = 'GradientCorrelation'

    def steepest_descent_images(self, image, dW_dp, forward=None):
        n_dims = image.n_dims

        # compute gradient
        # gradient:  height  x  width  x  (n_channels x n_dims)
        gradient_img = self._calculate_gradients(image, forward=forward)

        # reshape gradient
        # first_grad:  n_pixels  x  (n_channels x n_dims)
        first_grad = gradient_img.as_vector(keep_channels=True)

        # reshape gradient
        # gradient:  n_pixels  x  n_channels  x  n_dims
        first_grad = np.reshape(first_grad, (-1, image.n_channels,
                                             image.n_dims))

        # compute IGOs (remember axis 0 is y, axis 1 is x)
        # phi:       n_pixels  x  n_channels
        # _cos_phi:  n_pixels  x  n_channels
        # _sin_phi:  n_pixels  x  n_channels
        phi = np.angle(first_grad[..., 1] + first_grad[..., 0] * 1j)
        self._cos_phi = np.cos(phi)
        self._sin_phi = np.sin(phi)

        # concatenate sin and cos terms so that we can take the second
        # derivatives correctly. sin(phi) = y and cos(phi) = x which is the
        # correct ordering when multiplying against the warp Jacobian
        # _cos_phi:     n_pixels  x  n_channels
        # _sin_phi:     n_pixels  x  n_channels
        # angle_grads:  n_pixels  x  n_channels x  n_dims
        angle_grads = np.concatenate([self._sin_phi[..., None],
                                      self._cos_phi[..., None]], axis=2)

        # reshape angle_grads
        # gradient:  height  x  width  x  (n_channels x n_dims)
        angle_grads = np.reshape(angle_grads, (first_grad.shape[0], -1))
        angle_grads = gradient_img.from_vector(angle_grads)

        # compute IGOs gradient
        # second_grad:  height  x  width  x  (n_channels x n_dims x n_dims)
        second_grad = self._calculate_gradients(angle_grads)

        # reshape gradient
        # second_grad:  n_pixels  x  (n_channels x n_dims)
        second_grad = second_grad.as_vector(keep_channels=True).copy()

        # reshape IGOs gradient
        # second_grad:  n_pixels  x  n_channels  x  n_dims  x  n_dims
        second_grad = np.reshape(second_grad, (-1, image.n_channels,
                                               n_dims, n_dims))

        # Fix crossed derivatives: dydx = dxdy
        second_grad[:, :, 1, 0] = second_grad[:, :, 0, 1]

        # complete full IGOs gradient computation
        # second_grad:  n_pixels  x  n_channels  x  n_dims  x  n_dims
        second_grad[..., 1] = (-self._sin_phi[..., None] *
                               second_grad[..., 1])
        second_grad[..., 0] = (self._cos_phi[..., None] *
                               second_grad[..., 0])

        # compute steepest descent images
        # second_grad: n_pixels  x  n_channels  x            x n_dims x n_dims
        # dW_dp:       n_pixels  x              x  n_params  x n_dims x
        # sdi:         n_pixels  x  n_channels  x  n_params
        sdi = np.sum(np.sum(dW_dp[:, None, :, :, None] *
                            second_grad[:, :, None, :, :], axis=3), axis=3)

        # compute constant N
        # _N:  1
        self._N = np.product(image.shape)

        # reshape steepest descent images
        # sdi:  (n_pixels x n_channels)  x  n_params
        return np.reshape(sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, sdi):
        # compute hessian
        # sdi:      (n_pixels x n_channels x n_dims)  x  n_parameters
        # hessian:             n_parameters           x  n_parameters
        return sdi.T.dot(sdi)

    def steepest_descent_update(self, sdi, IWxp, template):
        # compute IWxp gradient
        # IWxp_gradient:  height  x  width  x  (n_channels x n_dims)
        IWxp_gradient = self._calculate_gradients(IWxp)

        # reshape IWxp gradient
        # IWxp_gradient:  n_pixels  x  (n_channels x n_dims)
        IWxp_gradient = IWxp_gradient.as_vector(keep_channels=True)

        # reshape IWxp first gradient
        # IWxp_gradient:  n_pixels  x  n_channels  x  n_dims
        IWxp_gradient = np.reshape(IWxp_gradient, (-1, IWxp.n_channels,
                                                   IWxp.n_dims))

        # compute IGOs (remember axis 0 is y, axis 1 is x)
        # phi:           n_pixels  x  n_channels
        # IWxp_cos_phi:  n_pixels  x  n_channels
        # IWxp_sin_phi:  n_pixels  x  n_channels
        phi = np.angle(IWxp_gradient[..., 1] + IWxp_gradient[..., 0] * 1j)
        IWxp_cos_phi = np.cos(phi)
        IWxp_sin_phi = np.sin(phi)

        # compute angular error
        # _error_img:  (n_pixels x n_channels)
        self._error_img = (self._cos_phi * IWxp_sin_phi -
                           self._sin_phi * IWxp_cos_phi).flatten()

        # compute steepest descent update
        # sdi:         (n_pixels x n_channels)  x  n_parameters
        # _error_img:  (n_pixels x n_channels)
        # sdu:             n_parameters
        sdu = sdi.T.dot(self._error_img)

        # compute step size
        # qp:  1
        # l:   1
        qp = np.sum(IWxp_cos_phi * self._cos_phi +
                    IWxp_sin_phi * self._sin_phi)
        l = self._N / qp

        # compute steepest descent update
        # l:                  1
        # sdu:           n_parameters
        # weighted_sdu:  n_parameters
        return l * sdu
