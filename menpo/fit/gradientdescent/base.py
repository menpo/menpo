from __future__ import division
import abc
import numpy as np
from scipy.stats import multivariate_normal

from menpo.fit.base import Fitter
from menpo.fit.fittingresult import SemiParametricFittingResult
from menpo.fitmultilevel.functions import build_sampling_grid


# TODO: document me
# TODO: see if residuals can be incorporated here
# TODO: deal with regularization/prior inside the transforms, this is also
# relevant for LucasKanade objects.
class GradientDescent(Fitter):
    r"""
    """
    def __init__(self, classifiers, patch_shape, transform,
                 eps=10**-10):
        self.classifiers = classifiers
        self.patch_shape = patch_shape
        self.transform = transform
        self.eps = eps
        # pre-computations
        self._set_up()

    def _create_fitting(self, image, parameters, gt_shape=None):
        return SemiParametricFittingResult(
            image, self, parameters=[parameters], gt_shape=gt_shape)

    def get_parameters(self, shape):
        self.transform.set_target(shape)
        return self.transform.as_vector()


# TODO: document me
class ActiveShapeModel(GradientDescent):

    @abc.abstractproperty
    def algorithm(self):
        return "ASM"

    def _set_up(self):
        raise ValueError("Not implemented yet")

    def _fit(self, fitting, max_iters=20):
        raise ValueError("Not implemented yet")


# TODO: document me
class ConvexQuadraticFitting(GradientDescent):

    @abc.abstractproperty
    def algorithm(self):
        return "CQF"

    def _set_up(self):
        raise ValueError("Not implemented yet")

    def _fit(self, fitting, max_iters=20):
        raise ValueError("Not implemented yet")


# TODO: document me
class RegularizedLandmarkMeanShift(GradientDescent):

    def algorithm(self):
        return "RLMS"

    def _set_up(self):
        self._sampling_grid = build_sampling_grid(self.patch_shape)

        # Gaussian-KDE
        mean = np.zeros(self.transform.n_dims)
        self._rho = 10 * self.transform.model.noise_variance
        mvn = multivariate_normal(mean=mean, cov=self._rho)
        self._kernel_grid = mvn.pdf(self._sampling_grid)

        # Transform Jacobian
        self._J = self.transform.jacobian([])

        # Prior
        augmented_eigenvalues = np.hstack((np.ones(4),
                                           self.transform.model.eigenvalues))
        self._J_regularizer = self._rho / augmented_eigenvalues
        # set uninformative prior for similarity weights
        self._J_regularizer[:4] = 0

        # Inverse Hessian
        self._inv_H = np.linalg.inv(np.diag(self._J_regularizer) +
                                    np.dot(self._J.T, self._J))

    def _fit(self, fitting, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting.image
        target = self.transform.target
        n_iters = 0

        max_x = image.shape[0] - 1
        max_y = image.shape[1] - 1

        image_pixels = np.reshape(image.pixels, (-1, image.n_channels))
        response_image = np.zeros((image.shape[0], image.shape[1],
                                   target.n_points))

        # Compute responses
        for j, clf in enumerate(self.classifiers):
            response_image[:, :, j] = np.reshape(clf(image_pixels),
                                                 image.shape)

        while n_iters < max_iters and error > self.eps:

            mean_shift_target = np.zeros((target.n_points, target.n_dims))

            # Compute mean-shift vectors
            for j, point in enumerate(target.points):

                patch_grid = (self._sampling_grid +
                              np.round(point[None, None, ...]).astype(int))

                x = patch_grid[:, :, 0]
                y = patch_grid[:, :, 1]

                # deal with boundaries
                x[x > max_x] = max_x
                y[y > max_y] = max_y
                x[x < 0] = 0
                y[y < 0] = 0

                kernel_response = response_image[x, y, j] * self._kernel_grid
                normalizer = np.sum(kernel_response)
                normalized_kernel_response = kernel_response / normalizer

                mean_shift_target[j, :] = np.sum(
                    normalized_kernel_response * (x, y), axis=(1, 2))

            # Compute parameter updates
            difference = np.dot(self._J.T,
                                mean_shift_target.ravel() - target.as_vector())
            regularizer = self._J_regularizer * self.transform.as_vector()
            delta_p = -np.dot(self._inv_H, regularizer - difference)

            # Update transform weights
            parameters = self.transform.as_vector() + delta_p
            fitting.parameters.append(parameters)
            self.transform.from_vector_inplace(parameters)
            target = self.transform.target

            # Test convergence
            error = np.abs(np.linalg.norm(delta_p))
            n_iters += 1

        fitting.fitted = True
        return fitting
