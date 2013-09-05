"""
This module contains a set of similarity measures that was designed for use
within the Lucas-Kanade framework. They therefore expose a number of methods
that make them useful for inverse compositional and forward additive
Lucas-Kanade.

These similarity measures are designed to be dimension independent where
possible. For this reason, some methods look more complicated than would be
normally the case. For example, calculating the Hessian involves summing
a multi-dimensional array, so we dynamically calculate the list of axes
to summer over. However, the basics of the logic, other than dimension
reduction, should be similar to the original algorithms.

References
----------

.. [1] Lucas, Bruce D., and Takeo Kanade.
       "An iterative image registration technique with an application to stereo
       vision."
       IJCAI. Vol. 81. 1981.
"""
import abc
import numpy as np
from numpy.fft import fftshift, fftn
import scipy.linalg
from pybug.convolution import log_gabor
from pybug.image import Image


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

        The input image is vectorised (``N``-pixels) so that masked images can
        be handled.

        Parameters
        ----------
        image : :class:`pybug.image.base.Image`
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
        IWxp : :class:`pybug.image.base.Image`
            Either the warped image or the template
            (depending on the framework)
        template : :class:`pybug.image.base.Image`
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

        If ``forward`` is provided, then the gradients are warped
        (as is required in the forward additive algorithm)

        Parameters
        ----------
        image : :class:`pybug.image.base.Image`
            The image to calculate the gradients for
        forward : (:class:`template <pybug.image.base.Image>`, :class:`template <pybug.transform.base.Transform>`, ``warp``), optional
            A tuple containing the extra parameters required for the function
            ``warp`` (which should be passed as a function handle).

            Default: ``None``
        """
        # Calculate the gradient over the image
        gradient = image.gradient()

        # Warp image for forward additive, if we've been given a transform
        if forward:
            template, transform, warp = forward
            gradient = warp(gradient, template, transform)

        return gradient


class LSIntensity(Residual):

    def steepest_descent_images(self, image, dW_dp, forward=None):
        # compute gradient
        # gradient:  height  x  width  x  n_channels
        gradient = self._calculate_gradients(image, forward=forward)

        # reshape gradient
        # gradient:  n_pixels  x  (n_channels x n_dims)
        gradient = gradient.as_vector(keep_channels=True)

        # reshape gradient
        # gradient:  n_pixels  x  n_channels  x  n_dims
        gradient = np.reshape(gradient, (gradient.shape[0], -1, 2))

        # compute steepest descent images
        # gradient:  n_pixels  x  n_channels  x  n_params  x  n_dims
        # dW_dp:     n_pixels  x  n_channels  x  n_params  x  n_dims
        # sdi:       n_pixels  x  n_channels  x  n_params
        sdi = np.sum(dW_dp[:, None, :, :] * gradient[:, :, None, :], axis=3)

        # reshape steepest descent images
        # sdi:  (n_pixels x n_channels)  x  n_params
        return np.reshape(sdi, (-1, dW_dp.shape[1]))

    def calculate_hessian(self, VT_dW_dp):
        return VT_dW_dp.T.dot(VT_dW_dp)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        self._error_img = IWxp.as_vector() - template.as_vector()
        return VT_dW_dp.T.dot(self._error_img)


class GaborFourier(Residual):

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
        gradient = self._calculate_gradients(image, forward=forward)
        gradient_vec = gradient.as_vector(keep_channels=True)
        VT_dW_dp = np.sum(dW_dp * gradient_vec[:, np.newaxis, ...], axis=2)

        # We have to take the FFT, therefore, we need an image
        # Reshape back to an image from the vectorized form. Use the gradient
        # shape in case of FA version (gradients get warped)
        sd_image_shape = gradient.shape + (VT_dW_dp.shape[-1],)
        sd_image = np.reshape(VT_dW_dp, sd_image_shape)
        fft_axes = range(image.n_dims)

        # Compute FFT over each parameter
        # Then, reshape back to vector for consistency with other residuals
        FT_VT_dW_dp = fftshift(fftn(sd_image, axes=fft_axes), axes=fft_axes)
        # Reshape to (n_pixels x n_params)
        return FT_VT_dW_dp.reshape([-1, VT_dW_dp.shape[-1]])

    def calculate_hessian(self, VT_dW_dp):
        filtered_jac = (self._filter_bank[..., None] ** 0.5) * VT_dW_dp
        return np.conjugate(filtered_jac).T.dot(filtered_jac)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        # Calculate FFT error image and flatten
        self._error_img = fftshift(fftn(np.squeeze(IWxp.pixels -
                                                   template.pixels))).flatten()
        ft_error_img = self._filter_bank * self._error_img
        return VT_dW_dp.T.dot(np.conjugate(ft_error_img))


class ECC(Residual):

    def __normalise_images(self, image):
        # TODO: do we need to copy the image?
        new_im = Image(image.pixels, mask=image.mask)
        i = new_im.pixels
        i -= np.mean(i)
        i /= scipy.linalg.norm(i)
        i = np.nan_to_num(i)

        new_im.pixels = i
        return new_im

    def steepest_descent_images(self, image, dW_dp, forward=None):
        norm_image = self.__normalise_images(image)

        gradient = self._calculate_gradients(norm_image,
                                             forward=forward)
        gradient = gradient.as_vector(keep_channels=True)
        return np.sum(dW_dp * gradient[:, np.newaxis, :], axis=2)

    def calculate_hessian(self, VT_dW_dp):
        H = VT_dW_dp.T.dot(VT_dW_dp)
        self._H_inv = scipy.linalg.inv(H)

        return H

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        normalised_IWxp = self.__normalise_images(IWxp).as_vector()
        normalised_template = self.__normalise_images(template).as_vector()

        Gt = VT_dW_dp.T.dot(normalised_template)
        Gw = VT_dW_dp.T.dot(normalised_IWxp)

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

        return VT_dW_dp.T.dot(self._error_img)


class GradientImages(Residual):

    def __regularise_gradients(self, gradients):
        pixels = gradients.pixels
        ab = np.sqrt(np.sum(np.square(pixels), -1))
        m_ab = np.median(ab)
        ab = ab + m_ab

        gradients.pixels = pixels / ab[..., None]
        return gradients

    def steepest_descent_images(self, image, dW_dp, forward=None):
        n_dims = image.n_dims
        gradient = self._calculate_gradients(image, forward=forward)

        self.__template_gradients = self.__regularise_gradients(gradient)

        # Calculate second order derivatives over each dimension
        gradient = self._calculate_gradients(self.__template_gradients)

        # Set the second derivatives that should theoretically match to by the
        # same value. For example, in 3D (xx means the second order derivative
        # of x with respect to x):
        #   xy = yx, xz = zx, yz = zy =>
        #   gradients[0][1] = gradients[1][0]
        #   gradients[0][2] = gradients[2][0]
        #   gradients[1][2] = gradients[2][1]
        # Therefore, we use exploit the symmetry to fix the derivatives:
        #   xx  xy  xz
        #   yx  yy  yz
        #   zx  zy  zz
        # Where we can see that the we need the upper triangle = lower
        # triangle
        gradient = gradient.as_vector(keep_channels=True).reshape([-1,
                                                                   n_dims,
                                                                   n_dims])
        mask = np.zeros([n_dims, n_dims], dtype=np.bool)
        # Find the lower triangle indices
        tri_indices = np.tril_indices(n_dims, k=-1)
        mask[tri_indices] = True
        gradient[:, mask] = gradient[:, mask.T]
        gradient = gradient.reshape([-1, n_dims ** 2])

        # Reshape to keep the xs and ys separate
        gradient = gradient.reshape([-1, n_dims, n_dims])
        sd = dW_dp[:, :, None, :] * gradient[:, None, ...]
        return np.sum(sd, axis=3)

    def calculate_hessian(self, VT_dW_dp):
        # Loop over every dimension and compute the individual Hessians, e.g:
        #   Hx = Gx.T * Gx
        # And then sum
        # Sum over the pixels and dimensions in order to yield a p x q matrix
        # n = number of pixels
        # p,q = number of parameters
        # d = number of dimensions
        return np.einsum('npd, nqd', VT_dW_dp, VT_dW_dp)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        IWxp_gradients = self._calculate_gradients(IWxp)
        IWxp_gradients = self.__regularise_gradients(IWxp_gradients)

        IWxp_pixels = IWxp_gradients.as_vector(keep_channels=True)
        T_pixels = self.__template_gradients.as_vector(keep_channels=True)

        error_imgs = IWxp_pixels - T_pixels
        self._error_img = np.sum(error_imgs, 1)

        # Compute steepest descent update for each dimension, e.g:
        # sd_x = Gx.T * error_img_x
        # Then sd_x + sd_y + ...
        # Sum over pixels and dimensions to leave just the parameter updates,
        # n = number of pixels
        # p = number of parameters
        # d = number of dimensions
        return np.einsum('npd, nd', VT_dW_dp, error_imgs)


class GradientCorrelation(Residual):

    def steepest_descent_images(self, image, dW_dp, forward=None):
        gradients = self._calculate_gradients(image, forward=forward)
        first_grad = gradients.as_vector(keep_channels=True)

        # Axis 0 = y, Axis 1 = x
        # Therefore, calculate the angle between the gradients
        phi = np.angle(first_grad[..., 1] + first_grad[..., 0] * 1j)
        self._cos_phi = np.cos(phi)
        self._sin_phi = np.sin(phi)

        # Concatenate the sin and cos so that we can take the second
        # derivatives correctly. sin(phi) = y and cos(phi) = x which is the
        # correct ordering when multiplying against the warp Jacobian
        angle_grads = np.concatenate([self._sin_phi[..., None],
                                      self._cos_phi[..., None]], axis=1)
        angle_grads = gradients.from_vector(angle_grads)
        second_grad = self._calculate_gradients(angle_grads).as_vector(
            keep_channels=True)
        # Fix the derivatives - yx = xy
        second_grad[..., 1] = second_grad[..., 2]

        # Jacobian of axis 1 values (x)
        G1 = np.sum(-self._sin_phi[..., None, None] *
                    dW_dp * second_grad[:, None, 2:], axis=2)

        # Jacobian of axis 0 values (y)
        G0 = np.sum(self._cos_phi[..., None, None] *
                    dW_dp * second_grad[:, None, :2], axis=2)

        self._N = np.product(image.shape)
        self._J = np.sum([G0, G1], axis=0)

        return G0, G1

    def calculate_hessian(self, VT_dW_dp):
        G0, G1 = VT_dW_dp
        G0 = G0.T.dot(G0)
        G1 = G1.T.dot(G1)

        return G0 + G1

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        IWxp_gradients = self._calculate_gradients(IWxp)
        IWxp_grads = IWxp_gradients.as_vector(keep_channels=True)

        # Axis 0 = y, Axis 1 = x
        # Therefore, calculate the angle between the gradients
        phi = np.angle(IWxp_grads[..., 1] + IWxp_grads[..., 0] * 1j)
        IWxp_cos_phi = np.cos(phi)
        IWxp_sin_phi = np.sin(phi)

        # Calculate the angular error
        ang_err = self._cos_phi * IWxp_sin_phi - self._sin_phi * IWxp_cos_phi
        self._error_img = ang_err

        # Calculate the step size
        JT_Sdelta = self._J.T.dot(ang_err)

        qp = IWxp_cos_phi * self._cos_phi + IWxp_sin_phi * self._sin_phi
        qp = np.sum(qp)

        l = self._N / qp
        image_error = l * JT_Sdelta

        return image_error
