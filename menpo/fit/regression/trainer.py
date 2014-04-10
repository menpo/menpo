from __future__ import division, print_function
import abc
import numpy as np

from menpo.fitmultilevel.functions import (noisy_align, build_sampling_grid,
                                           extract_local_patches)
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog
from menpo.fit.fittingresult import (NonParametricFittingResult,
                                     SemiParametricFittingResult,
                                     ParametricFittingResult)

from .base import (NonParametricRegressor, SemiParametricRegressor,
                   ParametricRegressor)
from .parametricfeatures import weights
from .regressionfunctions import regression, mlr


#TODO: document me
class RegressorTrainer(object):
    r"""
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, reference_shape, regression_type=mlr,
                 regression_features=None, noise_std=0.04, rotation=False,
                 n_perturbations=10):
        self.reference_shape = reference_shape
        self.regression_type = regression_type
        self.regression_features = regression_features
        self.rotation = rotation
        self.noise_std = noise_std
        self.n_perturbations = n_perturbations

    def _regression_data(self, images, gt_shapes, perturbed_shapes):

        n_images = len(images)

        features = []
        delta_ps = []
        for j, (i, s, p_shape) in enumerate(zip(images, gt_shapes,
                                                perturbed_shapes)):
            for ps in p_shape:
                features.append(self.features(i, ps))
                delta_ps.append(self.delta_ps(s, ps))
            print(' - {} % '.format(round(100*(j+1)/n_images)), end='\r')

        return np.asarray(features), np.asarray(delta_ps)

    @abc.abstractmethod
    def features(self, image, shape):
        pass

    @abc.abstractmethod
    def delta_ps(self, gt_shape, perturbed_shape):
        pass

    def train(self, images, shapes, perturbed_shapes=None, **kwargs):
        r"""
        """
        n_images = len(images)
        n_shapes = len(shapes)

        print('- generating regression data')
        if n_images != n_shapes:
            raise ValueError("The number of shapes must be equal to "
                             "the number of images.")

        elif not perturbed_shapes:
            perturbed_shapes = self.perturb_shapes(shapes)
            features, delta_ps = self._regression_data(
                images, shapes, perturbed_shapes)

        elif n_images == len(perturbed_shapes):
            features, delta_ps = self._regression_data(
                images, shapes, perturbed_shapes)

        else:
            raise ValueError("The number of perturbed shapes must be "
                             "equal or multiple to the number of images.")

        print('- performing regression')
        regressor = regression(features, delta_ps, self.regression_type,
                               **kwargs)

        print('- computing regression rmse')
        estimated_delta_ps = regressor(features)
        error = np.sqrt(np.mean(np.sum((delta_ps - estimated_delta_ps) ** 2,
                                       axis=1)))
        print(' - error = {}'.format(error))

        return self._build_regressor(regressor, self.features)

    def perturb_shapes(self, gt_shape):
        return [[self._perturb_shape(s) for _ in range(self.n_perturbations)]
                for s in gt_shape]

    def _perturb_shape(self, gt_shape):
        return noisy_align(self.reference_shape, gt_shape,
                           noise_std=self.noise_std
                           ).apply(self.reference_shape)

    @abc.abstractmethod
    def _build_regressor(self, regressor, features):
        pass


#TODO: document me
class NonParametricRegressorTrainer(RegressorTrainer):
    r"""
    """
    def __init__(self, reference_shape, regression_type=mlr,
                 regression_features=sparse_hog, patch_shape=(16, 16),
                 noise_std=0.04, rotation=False, n_perturbations=10):
        super(NonParametricRegressorTrainer, self).__init__(
            reference_shape, regression_type=regression_type,
            regression_features=regression_features, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.patch_shape = patch_shape
        self.sampling_grid = build_sampling_grid(patch_shape)

    def _create_fitting(self, image, shapes, gt_shape=None):
        return NonParametricFittingResult(image, self, shapes=[shapes],
                                          gt_shape=gt_shape)

    def features(self, image, shape):
        patches = extract_local_patches(image, shape, self.sampling_grid)
        features = [compute_features(p, self.regression_features).pixels.ravel()
                    for p in patches]
        return np.hstack((np.asarray(features).ravel(), 1))

    def delta_ps(self, gt_shape, perturbed_shape):
        return (gt_shape.as_vector() -
                perturbed_shape.as_vector())

    def _build_regressor(self, regressor, features):
        return NonParametricRegressor(regressor, features)


#TODO: Document me
class SemiParametricRegressorTrainer(NonParametricRegressorTrainer):
    r"""
    """
    def __init__(self, transform, reference_shape, regression_type=mlr,
                 regression_features=sparse_hog, patch_shape=(16, 16),
                 update='compositional', noise_std=0.04, rotation=False,
                 n_perturbations=10):
        super(SemiParametricRegressorTrainer, self).__init__(
            reference_shape, regression_type=regression_type,
            regression_features=regression_features, patch_shape=patch_shape,
            noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations)
        self.transform = transform
        self.update = update

    @property
    def algorithm(self):
        return "SemiParametric"

    def _create_fitting(self, image, shapes, gt_shape=None):
        return SemiParametricFittingResult(image, self, parameters=[shapes],
                                           gt_shape=gt_shape)

    def delta_ps(self, gt_shape, perturbed_shape):
        self.transform.target = gt_shape
        gt_ps = self.transform.as_vector()
        self.transform.target = perturbed_shape
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features, ):
        return SemiParametricRegressor(regressor, features, self.transform,
                                       self.update)


#TODO: Document me
class ParametricRegressorTrainer(RegressorTrainer):
    r"""
    """
    def __init__(self, appearance_model, transform, reference_shape,
                 regression_type=mlr, regression_features=weights,
                 update='compositional', noise_std=0.04, rotation=False,
                 n_perturbations=10, interpolator='scipy'):
        super(ParametricRegressorTrainer, self).__init__(
            reference_shape, regression_type=regression_type,
            regression_features=regression_features, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.appearance_model = appearance_model
        self.template = appearance_model.mean
        self.regression_features = regression_features
        self.transform = transform
        self.update = update
        self.interpolator = interpolator

    def _create_fitting(self, image, shapes, gt_shape=None):
        return ParametricFittingResult(image, self, parameters=[shapes],
                                       gt_shape=gt_shape)

    def features(self, image, shape):
        self.transform.set_target(shape)
        warped_image = image.warp_to(self.template.mask, self.transform,
                                     interpolator=self.interpolator)
        return np.hstack(
            (self.regression_features(self.appearance_model,
                                      warped_image), 1))

    def delta_ps(self, gt_shape, perturbed_shape):
        self.transform.set_target(gt_shape)
        gt_ps = self.transform.as_vector()
        self.transform.set_target(perturbed_shape)
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features):
        return ParametricRegressor(
            regressor, features, self.appearance_model, self.transform,
            self.update)


#TODO: Document me
class SemiParametricClassifierBasedRegressorTrainer(
        NonParametricRegressorTrainer):
    r"""
    """
    def __init__(self, classifiers, transform, reference_shape,
                 regression_type=mlr, patch_shape=(16, 16),
                 noise_std=0.04, rotation=False,
                 n_perturbations=10):
        super(SemiParametricClassifierBasedRegressorTrainer, self).__init__(
            reference_shape, regression_type=regression_type,
            patch_shape=patch_shape, noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations)
        self.classifiers = classifiers
        self.transform = transform
        self.update = 'additive'

    def features(self, image, shape):
        patches = extract_local_patches(image, shape, self.sampling_grid)
        features = [clf(np.reshape(p.pixels, (-1, p.n_channels)))
                    for (clf, p) in zip(self.classifiers, patches)]
        return np.hstack((np.asarray(features).ravel(), 1))

    def _create_fitting(self, image, shapes, gt_shape=None):
        return SemiParametricFittingResult(image, self, parameters=[shapes],
                                           gt_shape=gt_shape)

    def delta_ps(self, gt_shape, perturbed_shape):
        self.transform.set_target(gt_shape)
        gt_ps = self.transform.as_vector()
        self.transform.set_target(perturbed_shape)
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features, ):
        return SemiParametricRegressor(regressor, features, self.transform,
                                       self.update)
