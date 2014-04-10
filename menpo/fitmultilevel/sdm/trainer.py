from __future__ import division, print_function
import abc
import numpy as np

from menpo.transform import Scale, AlignmentSimilarity
from menpo.model.pdm import PDM, OrthoPDM
from menpo.transform.modeldriven import ModelDrivenTransform, OrthoMDTransform

from menpo.fit.regression.trainer import (
    NonParametricRegressorTrainer, ParametricRegressorTrainer,
    SemiParametricClassifierBasedRegressorTrainer)
from menpo.fit.regression.regressionfunctions import mlr
from menpo.fit.regression.parametricfeatures import weights
from menpo.fitmultilevel.functions import mean_pointcloud
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog

from .base import (SupervisedDescentMethodFitter, SupervisedDescentAAMFitter,
                   SupervisedDescentCLMFitter)


# TODO: document me
class SupervisedDescentTrainer(object):
    r"""
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, regression_type=mlr, regression_features=None,
                 feature_type=None, n_levels=3, downscale=2,
                 scaled_levels=True, noise_std=0.04, rotation=False,
                 n_perturbations=10, interpolator='scipy', **kwargs):
        self.regression_type = regression_type
        self.regression_features = regression_features
        self.feature_type = feature_type
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_levels = scaled_levels
        self.noise_std = noise_std
        self.rotation = rotation
        self.n_perturbations = n_perturbations
        self.interpolator = interpolator

    def train(self, images, group=None, label='all', **kwargs):
        r"""
        """
        print('- Computing reference shape')
        self.reference_shape = self._compute_reference_shape(images, group,
                                                             label)

        print('- Normalizing object size')
        self._rescale_reference_shape()
        images = [i.rescale_to_reference_shape(self.reference_shape,
                                               group=group, label=label,
                                               interpolator=self.interpolator)
                  for i in images]

        print('- Generating multilevel scale space')
        if self.scaled_levels:
            # Gaussian pyramid
            generator = [i.gaussian_pyramid(n_levels=self.n_levels,
                                            downscale=self.downscale)
                         for i in images]
        else:
            # Smoothing pyramid
            generator = [i.smoothing_pyramid(n_levels=self.n_levels,
                                             downscale=self.downscale)
                         for i in images]

        print('- Generating multilevel feature space')
        images = []
        for _ in np.arange(self.n_levels):
            images.append([compute_features(g.next(), self.feature_type)
                           for g in generator])
        images.reverse()

        print('- Extracting ground truth shapes')
        gt_shapes = [[i.landmarks[group][label].lms for i in img]
                     for img in images]

        print('- Building regressors')
        regressors = []
        # for each level
        for j, (level_images, level_gt_shapes) in enumerate(zip(images,
                                                                gt_shapes)):
            print(' - Level {}'.format(j))

            trainer = self._set_regressor_trainer(j)

            if j == 0:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          **kwargs)
            else:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          level_shapes, **kwargs)

            print(' - Generating next level data')

            level_shapes = trainer.perturb_shapes(gt_shapes[0])

            regressors.append(regressor)
            count = 0
            total = len(regressors) * len(images[0]) * len(level_shapes[0])
            for k, r in enumerate(regressors):

                test_images = images[k]
                test_gt_shapes = gt_shapes[k]

                fittings = []
                for (i, gt_s, level_s) in zip(test_images, test_gt_shapes,
                                              level_shapes):
                    fitting_sublist = []
                    for ls in level_s:
                        fitting = r.fit(i, ls)
                        fitting.gt_shape = gt_s
                        fitting_sublist.append(fitting)
                        count += 1

                    fittings.append(fitting_sublist)
                    print(' - {} % '.format(round(100*(count+1)/total)),
                          end='\r')

                if self.scaled_levels:
                    level_shapes = [[Scale(self.downscale,
                                           n_dims=self.reference_shape.n_dims
                                           ).apply(f.final_shape)
                                     for f in fitting_sublist]
                                    for fitting_sublist in fittings]
                else:
                    level_shapes = [[f.final_shape for f in fitting_sublist]
                                    for fitting_sublist in fittings]

            mean_error = np.mean(np.array([f.final_error
                                           for fitting_sublist in fittings
                                           for f in fitting_sublist]))
            print(' - Mean error = {}'.format(mean_error))

        return self._build_supervised_descent_fitter(regressors)

    @abc.abstractmethod
    def _compute_reference_shape(self, images, group, label):
        r"""
        """
        pass

    def _rescale_reference_shape(self):
        r"""
        """
        pass

    @abc.abstractmethod
    def _set_regressor_trainer(self, **kwargs):
        r"""
        """
        pass

    @abc.abstractmethod
    def _build_supervised_descent_fitter(self, regressors):
        pass


#TODO: Document me
class SupervisedDescentMethodTrainer(SupervisedDescentTrainer):
    r"""
    """
    def __init__(self, regression_type=mlr, regression_features=sparse_hog,
                 patch_shape=(16, 16), feature_type=None, n_levels=3,
                 downscale=1.5, scaled_levels=True, noise_std=0.04,
                 rotation=False, n_perturbations=10, diagonal_range=None,
                 interpolator='scipy'):
        super(SupervisedDescentMethodTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            feature_type=feature_type, n_levels=n_levels,
            downscale=downscale, scaled_levels=scaled_levels,
            noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations, interpolator=interpolator)
        self.patch_shape = patch_shape
        self.diagonal_range = diagonal_range

    def _compute_reference_shape(self, images, group, label):
        shapes = [i.landmarks[group][label].lms for i in images]
        return mean_pointcloud(shapes)

    def _rescale_reference_shape(self):
        if self.diagonal_range:
            x, y = self.reference_shape.range()
            scale = self.diagonal_range / np.sqrt(x**2 + y**2)
            Scale(scale, self.reference_shape.n_dims
                  ).apply_inplace(self.reference_shape)

    def _set_regressor_trainer(self, level):
        return NonParametricRegressorTrainer(
            self.reference_shape, regression_type=self.regression_type,
            regression_features=self.regression_features,
            patch_shape=self.patch_shape, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        return SupervisedDescentMethodFitter(
            regressors, self.feature_type, self.reference_shape,
            self.downscale, self.scaled_levels, self.interpolator)


#TODO: Document me
class SupervisedDescentAAMTrainer(SupervisedDescentTrainer):
    r"""
    """
    def __init__(self, aam, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                 update='compositional', md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None):
        super(SupervisedDescentAAMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            feature_type=aam.feature_type, n_levels=aam.n_levels,
            downscale=aam.downscale, scaled_levels=aam.scaled_levels,
            noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations, interpolator=aam.interpolator)
        self.aam = aam
        self.update = update
        self.md_transform = md_transform
        self.global_transform = global_transform

        if n_shape is not None:
            if type(n_shape) is int:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) is 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) is self.aam.n_levels:
                for sm, n in zip(self.aam.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be integer, integer list '
                                 'containing 1 or {} elements or '
                                 'None'.format(self.aam.n_levels))

        if n_appearance is not None:
            if type(n_appearance) is int:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) is 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) is self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_shape):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be integer, integer list '
                                 'containing 1 or {} elements or '
                                 'None'.format(self.aam.n_levels))

    def _compute_reference_shape(self, images, group, label):
        return self.aam.reference_shape

    def _normalize_object_size(self, images, group, label):
        return [i.rescale_to_reference_shape(self.reference_shape,
                                             group=group, label=label,
                                             interpolator=self.interpolator)
                for i in images]

    def _set_regressor_trainer(self, level):
        am = self.aam.appearance_models[level]
        sm = self.aam.shape_models[level]

        if self.md_transform is not ModelDrivenTransform:
            md_transform = self.md_transform(
                sm, self.aam.transform, self.global_transform,
                source=am.mean.landmarks['source'].lms)
        else:
            md_transform = self.md_transform(
                sm, self.aam.transform,
                source=am.mean.landmarks['source'].lms)

        return ParametricRegressorTrainer(
            am, md_transform, self.reference_shape,
            regression_type=self.regression_type, regression_features=self
            .regression_features, update=self.update,
            noise_std=self.noise_std, rotation=self.rotation,
            n_perturbations=self.n_perturbations,
            interpolator=self.interpolator)

    def _build_supervised_descent_fitter(self, regressors):
        return SupervisedDescentAAMFitter(self.aam, regressors)


#TODO: Document me
#TODO: Finish me
class SupervisedDescentCLMTrainer(SupervisedDescentTrainer):
    r"""
    """
    def __init__(self, clm, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                pdm_transform=OrthoPDM,
                global_transform=AlignmentSimilarity, n_shape=None):
        super(SupervisedDescentCLMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            feature_type=clm.feature_type, n_levels=clm.n_levels,
            downscale=clm.downscale, scaled_levels=clm.scaled_levels,
            noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations, interpolator=clm.interpolator)
        self.clm = clm
        self.patch_shape = clm.patch_shape
        self.pdm_transform = pdm_transform
        self.global_transform = global_transform

        if n_shape is not None:
            if type(n_shape) is int:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) is 1 and self.clm.n_levels > 1:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) is self.clm.n_levels:
                for sm, n in zip(self.clm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be integer, integer list '
                                 'containing 1 or {} elements or '
                                 'None'.format(self.clm.n_levels))

    def _compute_reference_shape(self, images, group, label):
        return self.clm.reference_shape

    #TODO: Finish me
    def _set_regressor_trainer(self, level):
        clfs = self.clm.classifiers[level]
        sm = self.clm.shape_models[level]

        if self.pdm_transform is not PDM:
            pdm_transform = self.pdm_transform(sm, self.global_transform)
        else:
            pdm_transform = self.pdm_transform(sm)

        return SemiParametricClassifierBasedRegressorTrainer(
            clfs, pdm_transform, self.reference_shape,
            regression_type=self.regression_type,
            patch_shape=self.patch_shape,
            noise_std=self.noise_std, rotation=self.rotation,
            n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        return SupervisedDescentCLMFitter(self.clm, regressors)
