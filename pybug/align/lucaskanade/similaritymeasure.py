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

Citations:

Lucas, Bruce D., and Takeo Kanade.
"An iterative image registration technique with an application to stereo
vision."
IJCAI. Vol. 81. 1981.
"""
import abc
import numpy as np
from numpy.fft import fftshift, fftn
import scipy.ndimage
import scipy.linalg
from pybug.convolution import log_gabor
import pybug.matlab as matlab
from pybug.warp import warp
from pybug.warp.base import map_coordinates_interpolator


class SimilarityMeasure(object):
    """
    An abstract base class for calculating the similarity between two images
    within the Lucas-Kanade algorithm. The classes were designed
    specifically to work within the Lucas-Kanade framework and so no
    guarantee is made that calling methods on these subclasses will generate
    correct results.
    """
    __metaclass__ = abc.ABCMeta

    @property
    def error(self):
        r"""
        The RMS of the error image. Will only generate a result if the
        steepest descent update has been calculated prior.

        .. math::
            error = \sqrt{\sum_x E(x)^2}

        where :math:`E(x) = T(x) - I(W(x;p))` within the forward additive
        framework.
        """
        return np.sqrt(np.mean(self._error_img ** 2))

    @abc.abstractmethod
    def steepest_descent_images(self, image, dW_dp, **kwargs):
        r"""
        Calculates the standard steepest descent images. Within the forward
        additive framework this is defined as

        .. math::
             \nabla I \frac{\partial W}{\partial p}

        :param image: The image to calculate the steepest descent
            images from, could be either the template or input image depending
            on which framework is used.
        :type image: ndarray
        :param dW_dp: The Jacobian of the warp.
        :type dW_dp: ndarray
        :return: The steepest descent images of shape
            (num_params x image_height x image_width)
        :rtype: ndarray
        """
        pass

    @abc.abstractmethod
    def calculate_hessian(self, VT_dW_dp):
        pass

    @abc.abstractmethod
    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        pass

    def _calculate_gradients(self, image, shape, transform=None,
                             interpolator=map_coordinates_interpolator):
        # Calculate the gradient over the image
        gradient = matlab.gradient(image)

        # Warp image for forward additive, if we've been given a transform
        if not transform is None:
            gradient = [warp(g, shape, transform, interpolator=interpolator)
                        for g in gradient]

        return gradient

    def _sum_over_axes(self, tensor, axes):
        tensor_summed = np.apply_over_axes(np.sum, tensor, axes)
        return np.squeeze(tensor_summed)

    def _sum_Hessian(self, H):
        # Creates a reverse list from n_dim + 1:2 to sum over
        # eg. for 3 dimensional image: [4, 3, 2]
        axes = range(len(H.shape) - 1, 1, -1)
        return self._sum_over_axes(H, axes)

    def _sum_steepest_descent(self, sd):
        # Creates a reverse list from n_dim:1 to sum over
        # eg. for 3 dimensional image: [3, 2, 1]
        axes = range(len(sd.shape) - 1, 0, -1)
        return self._sum_over_axes(sd, axes)


class LeastSquares(SimilarityMeasure):

    def steepest_descent_images(self, image, dW_dp, **kwargs):
        gradient = self._calculate_gradients(image, dW_dp.shape[-2:],
                                             **kwargs)

        # Add an extra axis for broadcasting
        gradient = [g[np.newaxis, ...] for g in gradient]
        # Concatenate gradient list into a vector
        gradient = np.concatenate(gradient, axis=0)

        return np.sum(dW_dp * gradient[:, np.newaxis, ...], axis=0)

    def calculate_hessian(self, VT_dW_dp):
        H = VT_dW_dp[:, np.newaxis, ...] * VT_dW_dp
        return self._sum_Hessian(H)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        self._error_img = IWxp - template
        sd = VT_dW_dp * self._error_img
        return self._sum_steepest_descent(sd)


class GaborFourier(SimilarityMeasure):

    def __init__(self, image_shape, **kwargs):
        if 'filter_bank' in kwargs:
            self._filter_bank = kwargs.get('filter_bank')
            if self._filter_bank.shape != image_shape:
                raise ValueError('Filter bank must match the '
                                 'size of the image')
        else:
            gabor = log_gabor(np.ones(image_shape), **kwargs)
            self._filter_bank = gabor[2]  # Get filter bank matrix

    def steepest_descent_images(self, image, dW_dp, **kwargs):
        gradient = self._calculate_gradients(image, dW_dp.shape[-2:],
                                             **kwargs)
        gradient = [g[np.newaxis, ...] for g in gradient]
        gradient = np.concatenate(gradient, axis=0)
        VT_dW_dp = np.sum(dW_dp * gradient[:, np.newaxis, ...], axis=0)

        # Get a range from 1 to number of image dimensions for computing
        # FFT over
        image_dims = range(1, len(image.shape) + 1)

        # Compute FFT over each parameter
        return fftshift(fftn(VT_dW_dp, axes=image_dims), axes=image_dims)

    def calculate_hessian(self, VT_dW_dp):
        filtered_jac = (self._filter_bank ** 0.5) * VT_dW_dp
        H = np.conjugate(filtered_jac[:, np.newaxis, ...]) * filtered_jac
        return self._sum_Hessian(H)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        self._error_img = fftshift(fftn(IWxp - template))
        ft_error_img = self._filter_bank * self._error_img
        sd = VT_dW_dp * np.conjugate(ft_error_img)
        return self._sum_steepest_descent(sd)


class ECC(SimilarityMeasure):

    def __normalise_images(self, image):
        # TODO: do we need to copy the image?
        i = np.copy(image)
        i -= np.mean(i)
        i /= scipy.linalg.norm(i)
        i = np.nan_to_num(i)

        return i

    def steepest_descent_images(self, image, dW_dp, **kwargs):

        norm_image = self.__normalise_images(image)

        gradient = self._calculate_gradients(norm_image, dW_dp.shape[-2:],
                                             **kwargs)
        gradient = [g[np.newaxis, ...] for g in gradient]
        gradient = np.concatenate(gradient, axis=0)
        G = np.sum(dW_dp * gradient[:, np.newaxis, ...], axis=0)

        return G

    def calculate_hessian(self, VT_dW_dp):
        H = VT_dW_dp[:, np.newaxis, ...] * VT_dW_dp
        H = self._sum_Hessian(H)
        self.__H_inv = scipy.linalg.inv(H)

        return H

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        normalised_IWxp = self.__normalise_images(IWxp)
        normalised_template = self.__normalise_images(template)

        Gt = self._sum_steepest_descent(VT_dW_dp * normalised_template)
        Gw = self._sum_steepest_descent(VT_dW_dp * normalised_IWxp)

        # Calculate the numerator
        IWxp_norm = scipy.linalg.norm(normalised_IWxp)
        num = (IWxp_norm ** 2) - np.dot(Gw.T, np.dot(self.__H_inv, Gw))

        # Calculate the denominator
        den1 = np.dot(normalised_template.flatten(), normalised_IWxp.flatten())
        den2 = np.dot(Gt.T, np.dot(self.__H_inv, Gw))
        den = den1 - den2

        # Calculate lambda to choose the step size
        # Avoid division by zero
        if den > 0:
            l = num / den
        else:
            # TODO: Should be other step described in paper
            l = 0

        self._error_img = l * normalised_IWxp - normalised_template
        Ge = VT_dW_dp * self._error_img

        return self._sum_steepest_descent(Ge)


class GradientImages(SimilarityMeasure):

    def __regularise_gradients(self, gradients):
        ab = np.sqrt(sum([np.square(g) for g in gradients]))
        m_ab = np.median(ab)
        ab = ab + m_ab
        return gradients / ab

    def steepest_descent_images(self, image, dW_dp, **kwargs):
        n_dim = len(image.shape)
        gradients = self._calculate_gradients(image, dW_dp.shape[-2:],
                                              **kwargs)

        self.__template_gradients = self.__regularise_gradients(gradients)

        # Calculate second order derivatives over each dimension
        gradients = [matlab.gradient(g) for g in self.__template_gradients]

        # Set the second derivatives that should theoretically match to by the
        # same value. For example, in 3D (xx means the second order derivative
        # of x with respect to x):
        #   xy = yx, xz = zx, yz = zy =>
        #   gradients[0][1] = gradients[1][0],
        #   gradients[0][2] = gradients[2][0],
        #   gradients[1][2] = gradients[2][1]
        for i in xrange(n_dim - 1):
            for j in xrange(i + 1, n_dim):
                gradients[i][j] = gradients[j][i]

        sd_images = []
        for g in gradients:
            g = [gg[np.newaxis, ...] for gg in g]
            g = np.concatenate(g, axis=0)
            sd = np.sum(dW_dp * g[:, np.newaxis, ...], axis=0)
            sd_images.append(sd)

        return sd_images

    def calculate_hessian(self, VT_dW_dp):
        # Loop over every dimension and compute the individual Hessians, e.g:
        # Hx = Gx.T * Gx
        H = [self._sum_Hessian(g[:, np.newaxis, ...] * g) for g in VT_dW_dp]

        # Hx + Hy ...
        return sum(H)

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        IWxp_gradients = matlab.gradient(IWxp)
        IWxp_gradients = self.__regularise_gradients(IWxp_gradients)

        error_imgs = [i - j for i, j in
                      zip(IWxp_gradients, self.__template_gradients)]
        self._error_img = sum(error_imgs)

        # Compute steepest descent update for each dimension, e.g:
        # sd_x = Gx.T * error_img_x
        G = [self._sum_steepest_descent(error_im * gradient)
             for gradient, error_im in zip(VT_dW_dp, error_imgs)]

        # sd_x + sd_y ...
        return sum(G)


class GradientCorrelation(SimilarityMeasure):

    def steepest_descent_images(self, image, dW_dp, **kwargs):
        gradients = self._calculate_gradients(image, dW_dp.shape[-2:],
                                              **kwargs)

        phi = np.angle(gradients[0] + gradients[1] * 1j)
        self.__cos_phi = np.cos(phi)
        self.__sin_phi = np.sin(phi)

        gradient_xs = matlab.gradient(self.__cos_phi)
        gradient_ys = matlab.gradient(self.__sin_phi)
        gradient_ys[0] = gradient_xs[1]

        # Jacobian of x values
        gradient_xs = [g[np.newaxis, ...] for g in gradient_xs]
        gradient_xs = np.concatenate(gradient_xs, axis=0)
        Gx = np.sum(-self.__sin_phi[np.newaxis, np.newaxis, ...] *
                    dW_dp * gradient_xs[:, np.newaxis, ...], axis=0)

        # Jacobian of y values
        gradient_ys = [g[np.newaxis, ...] for g in gradient_ys]
        gradient_ys = np.concatenate(gradient_ys, axis=0)
        Gy = np.sum(self.__cos_phi[np.newaxis, np.newaxis, ...] *
                    dW_dp * gradient_ys[:, np.newaxis, ...], axis=0)

        self.__N = image.size
        self.__J = np.sum([Gx, Gy], axis=0)

        return Gx, Gy

    def calculate_hessian(self, VT_dW_dp):
        Gx, Gy = VT_dW_dp
        Gx = np.sum(np.sum(Gx[:, np.newaxis, ...] * Gx, axis=2), axis=2)
        Gy = np.sum(np.sum(Gy[:, np.newaxis, ...] * Gy, axis=2), axis=2)

        return Gx + Gy

    def steepest_descent_update(self, VT_dW_dp, IWxp, template):
        IWxp_gradients = matlab.gradient(IWxp)

        # Calculate the gradient direction for the input image
        phi = np.angle(IWxp_gradients[0] + IWxp_gradients[1] * 1j)
        IWxp_cos_phi = np.cos(phi)
        IWxp_sin_phi = np.sin(phi)

        # Calculate the angular error
        ang_err = self.__cos_phi * IWxp_sin_phi - self.__sin_phi * IWxp_cos_phi
        self._error_img = ang_err

        # Calculate the step size
        JT_Sdelta = self.__J * ang_err[np.newaxis, ...]
        JT_Sdelta = np.sum(np.sum(JT_Sdelta, axis=1), axis=1)

        qp = IWxp_cos_phi * self.__cos_phi + IWxp_sin_phi * self.__sin_phi
        qp = np.sum(qp)

        l = self.__N / qp
        image_error = l * JT_Sdelta

        return image_error