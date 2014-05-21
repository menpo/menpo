from __future__ import division, print_function
import abc
import numpy as np

from menpo.fitmultilevel.functions import (noisy_align, build_sampling_grid,
                                           extract_local_patches)
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog
from menpo.fit.fittingresult import (NonParametricFittingResult,
                                     SemiParametricFittingResult,
                                     ParametricFittingResult)
from menpo.visualize import print_dynamic, progress_bar_str

from .base import (NonParametricRegressor, SemiParametricRegressor,
                   ParametricRegressor)
from .parametricfeatures import weights
from .regressionfunctions import regression, mlr


class RegressorTrainer(object):
    r"""
    An abstract base class for training regressors.

    Parameters
    ----------
    reference_shape: PointCloud
        The reference shape that will be used.
    regression_type: function/closure, Optional
        A function/closure that defines the regression technique to be used.
        Examples of such closures can be found in
        `menpo.fit.regression.regressionfunctions.py`

        Default: mlr
    regression_features: None or string or function/closure, Optional
        The features that are used during the regression.

        Default: None
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

        Default: False
    n_perturbations: int, Optional
        Defines the number of perturbations that will be applied to the
        training shapes.

        Default: 10
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

    def _regression_data(self, images, gt_shapes, perturbed_shapes,
                         verbose=False):
        r"""
        Method that generates the regression data: features and delta_ps.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images.
        gt_shapes: :class:`menpo.shape.PointCloud` list
            List of the ground truth shapes that correspond to the images.
        perturbed_shapes: :class:`menpo.shape.PointCloud` list
            List of the perturbed shapes in order to regress.
        verbose: boolean, optional
            If True, the progress is printed.

            Default: False
        """
        if verbose:
            print_dynamic('- Generating regression data')

        n_images = len(images)
        features = []
        delta_ps = []
        for j, (i, s, p_shape) in enumerate(zip(images, gt_shapes,
                                                perturbed_shapes)):
            for ps in p_shape:
                features.append(self.features(i, ps))
                delta_ps.append(self.delta_ps(s, ps))
            if verbose:
                print_dynamic('- Generating regression data - {}'.format(
                    progress_bar_str((j + 1.) / n_images, show_bar=False)))
        return np.asarray(features), np.asarray(delta_ps)

    @abc.abstractmethod
    def features(self, image, shape):
        r"""
        Abstract method to generate the features for the regression.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image..
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        """
        pass

    @abc.abstractmethod
    def delta_ps(self, gt_shape, perturbed_shape):
        r"""
        Abstract method to generate the delta_ps for the regression.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        perturbed_shape: :class:`menpo.shape.PointCloud`
            The perturbed shape.
        """
        pass

    def train(self, images, shapes, perturbed_shapes=None, verbose=False,
              **kwargs):
        r"""
        Trains a Regressor given a list of landmarked images.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images from which to train the regressor.
        shapes: :class:`menpo.shape.PointCloud` list
            List of the shapes that correspond to the images.
        perturbed_shapes: :class:`menpo.shape.PointCloud` list, optional
            List of the perturbed shapes used for the regressor training.

            Default: None
        verbose: bool, Optional
            Flag that controls information and progress printing.

            Default: False

        Returns
        -------
        regressor: `menpo.fit.regression.base`
            A regressor object.

        Raises
        ------
        ValueError
            The number of shapes must be equal to the number of images.
        ValueError
            The number of perturbed shapes must be equal or multiple to
            the number of images.
        """
        n_images = len(images)
        n_shapes = len(shapes)

        # generate regression data
        if n_images != n_shapes:
            raise ValueError("The number of shapes must be equal to "
                             "the number of images.")
        elif not perturbed_shapes:
            perturbed_shapes = self.perturb_shapes(shapes)
            features, delta_ps = self._regression_data(
                images, shapes, perturbed_shapes, verbose=verbose)
        elif n_images == len(perturbed_shapes):
            features, delta_ps = self._regression_data(
                images, shapes, perturbed_shapes, verbose=verbose)
        else:
            raise ValueError("The number of perturbed shapes must be "
                             "equal or multiple to the number of images.")

         # perform regression
        if verbose:
            print_dynamic('- Performing regression')
        regressor = regression(features, delta_ps, self.regression_type,
                               **kwargs)

        # compute regressor RMSE
        estimated_delta_ps = regressor(features)
        error = np.sqrt(np.mean(np.sum((delta_ps - estimated_delta_ps) ** 2,
                                       axis=1)))
        if verbose:
            print_dynamic('- Regression RMSE is {0:.5f}.\n'.format(error))
        return self._build_regressor(regressor, self.features)

    def perturb_shapes(self, gt_shape):
        r"""
        Perturbs the given shapes. The number of perturbations is defined by
        self.n_perturbations.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud` list
            List of the shapes that correspond to the images.will be perturbed.

        Returns
        -------
        perturbed_shapes: :class:`menpo.shape.PointCloud` list
            List of the perturbed shapes.
        """
        return [[self._perturb_shape(s) for _ in range(self.n_perturbations)]
                for s in gt_shape]

    def _perturb_shape(self, gt_shape):
        r"""
        Method that performs noisy alignment between the given ground truth
        shape and the reference shape.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        """
        return noisy_align(self.reference_shape, gt_shape,
                           noise_std=self.noise_std
                           ).apply(self.reference_shape)

    @abc.abstractmethod
    def _build_regressor(self, regressor, features):
        r"""
        Abstract method to build a regressor model.
        """
        pass


class NonParametricRegressorTrainer(RegressorTrainer):
    r"""
    Class for training a Non-Parametric Regressor.

    Parameters
    ----------
    reference_shape: PointCloud
        The reference shape that will be used.
    regression_type: function/closure, Optional
        A function/closure that defines the regression technique to be used.
        Examples of such closures can be found in
        `menpo.fit.regression.regressionfunctions.py`

        Default: mlr
    regression_features: None or string or function/closure, Optional
        The features that are used during the regression.

        Default: sparse_hog
    patch_shape: tuple, Optional
        The shape of the patches that will be extracted.

        Default: (16, 16)
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

        Default: False
    n_perturbations: int, Optional
        Defines the number of perturbations that will be applied to the
        training shapes.

        Default: 10
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

<<<<<<< HEAD
        # work out feature length
        patch = np.random.randn(patch_shape[0], patch_shape[1], 1)
        self.feature_length = regression_features(patch).ravel().shape[0]
=======
    @property
    def algorithm(self):
        r"""
        Returns the algorithm name.
        """
        return "Non-Parametric"
>>>>>>> master

    def _create_fitting(self, image, shapes, gt_shape=None):
        r"""
        Method that creates the fitting result object.

        Parameters
        ----------
        image: `menpo.image.MaskedImage`
            The image object.
        shapes: `menpo.shape.PointCloud` list
            The shapes.
        gt_shape: `menpo.shape.PointCloud`
            The ground truth shape.
        """
        return NonParametricFittingResult(image, self, shapes=[shapes],
                                          gt_shape=gt_shape)

    def features(self, image, shape):
<<<<<<< HEAD
        max_x = image.shape[0] - 1
        max_y = image.shape[1] - 1
        # compute sampling points
        grids = (self.sampling_grid[None, ...] +
                 np.round(shape.points).astype(int)[:, None, None, ...])
        xs = grids[:, :, :, 0]
        ys = grids[:, :, :, 1]
        # deal with boundaries
        xs[xs > max_x] = max_x
        ys[ys > max_y] = max_y
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        # compute features
        features = np.zeros((shape.n_points, self.feature_length))
        for j, (x, y) in enumerate(zip(xs, ys)):
            features[j, :] = self.regression_features(
                image.pixels[x, y, :]).ravel()
        # attach bias term
        return np.hstack((features.ravel(), 1))
=======
        r"""
        Method that extracts the features for the regression, which in this
        case are patch based.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image.
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        """
        patches = extract_local_patches(image, shape, self.sampling_grid)
        features = [compute_features(p,
                                     self.regression_features).pixels.ravel()
                    for p in patches]
        return np.hstack((np.asarray(features).ravel(), 1))
>>>>>>> master

    def delta_ps(self, gt_shape, perturbed_shape):
        r"""
        Method to generate the delta_ps for the regression.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        perturbed_shape: :class:`menpo.shape.PointCloud`
            The perturbed shape.
        """
        return (gt_shape.as_vector() -
                perturbed_shape.as_vector())

    def _build_regressor(self, regressor, features):
        r"""
        Method to build the NonParametricRegressor regressor object.
        """
        return NonParametricRegressor(regressor, features)


class SemiParametricRegressorTrainer(NonParametricRegressorTrainer):
    r"""
    Class for training a Semi-Parametric Regressor.

    Parameters
    ----------
    reference_shape: PointCloud
        The reference shape that will be used.
    regression_type: function/closure, Optional
        A function/closure that defines the regression technique to be used.
        Examples of such closures can be found in
        `menpo.fit.regression.regressionfunctions.py`

        Default: mlr
    regression_features: None or string or function/closure, Optional
        The features that are used during the regression.

        Default: sparse_hog
    patch_shape: tuple, Optional
        The shape of the patches that will be extracted.

        Default: (16, 16)
    update: 'compositional' or 'additive'
        Defines the way to update the warp.

        Default: 'compositional'
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

        Default: False
    n_perturbations: int, Optional
        Defines the number of perturbations that will be applied to the
        training shapes.

        Default: 10
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
        r"""
        Returns the algorithm name.
        """
        return "Semi-Parametric"

    def _create_fitting(self, image, shapes, gt_shape=None):
        r"""
        Method that creates the fitting result object.

        Parameters
        ----------
        image: `menpo.image.MaskedImage`
            The image object.
        shapes: `menpo.shape.PointCloud` list
            The shapes.
        gt_shape: `menpo.shape.PointCloud`
            The ground truth shape.
        """
        return SemiParametricFittingResult(image, self, parameters=[shapes],
                                           gt_shape=gt_shape)

    def delta_ps(self, gt_shape, perturbed_shape):
        r"""
        Method to generate the delta_ps for the regression.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        perturbed_shape: :class:`menpo.shape.PointCloud`
            The perturbed shape.
        """
        self.transform.target = gt_shape
        gt_ps = self.transform.as_vector()
        self.transform.target = perturbed_shape
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features):
        r"""
        Method to build the NonParametricRegressor regressor object.
        """
        return SemiParametricRegressor(regressor, features, self.transform,
                                       self.update)


class ParametricRegressorTrainer(RegressorTrainer):
    r"""
    Class for training a Parametric Regressor.

    Parameters
    ----------
    appearance_model: `menpo.model.pca`
        The appearance model to be used.
    transform: `model.transform.affine`
        The transform used for warping.
    reference_shape: PointCloud
        The reference shape that will be used.
    regression_type: function/closure, Optional
        A function/closure that defines the regression technique to be used.
        Examples of such closures can be found in
        `menpo.fit.regression.regressionfunctions.py`

        Default: mlr
    regression_features: None or function/closure or list of those, Optional
        The features that are used in the regressor.
        If list of length {aam.n_levels}, it specifies the feature to be used
        per level.
        If list of length 1, the specified feature will be used for all levels.

        Per level:
        Since the regressor in use is a Parametric one, these features
        can only come from:
        `menpo.fit.regression.parametricfeatures`

        If function/closure, the specified funtion will be used.
        If None, 'weights' will be used.

        Default: weights
    patch_shape: tuple, Optional
        The shape of the patches that will be extracted.

        Default: (16, 16)
    update: 'compositional' or 'additive'
        Defines the way to update the warp.

        Default: 'compositional'
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

        Default: False
    n_perturbations: int, Optional
        Defines the number of perturbations that will be applied to the
        training shapes.

        Default: 10
    interpolator: string
        Specifies the interpolator used in warping.

        Default: 'scipy'
    """
    def __init__(self, appearance_model, transform, reference_shape,
                 regression_type=mlr, regression_features=weights,
                 update='compositional', noise_std=0.04, rotation=False,
                 n_perturbations=10, interpolator='scipy'):
        if regression_features is None:
            regression_features = weights
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

    @property
    def algorithm(self):
        r"""
        Returns the algorithm name.
        """
        return "Parametric"

    def _create_fitting(self, image, shapes, gt_shape=None):
        r"""
        Method that creates the fitting result object.

        Parameters
        ----------
        image: `menpo.image.MaskedImage`
            The image object.
        shapes: `menpo.shape.PointCloud` list
            The shapes.
        gt_shape: `menpo.shape.PointCloud`
            The ground truth shape.
        """
        return ParametricFittingResult(image, self, parameters=[shapes],
                                       gt_shape=gt_shape)

    def features(self, image, shape):
        r"""
        Method that extracts the features for the regression, which in this
        case are patch based.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image.
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        """
        self.transform.set_target(shape)
        warped_image = image.warp_to(self.template.mask, self.transform,
                                     interpolator=self.interpolator)
        return np.hstack(
            (self.regression_features(self.appearance_model,
                                      warped_image), 1))

    def delta_ps(self, gt_shape, perturbed_shape):
        r"""
        Method to generate the delta_ps for the regression.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        perturbed_shape: :class:`menpo.shape.PointCloud`
            The perturbed shape.
        """
        self.transform.set_target(gt_shape)
        gt_ps = self.transform.as_vector()
        self.transform.set_target(perturbed_shape)
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features):
        r"""
        Method to build the NonParametricRegressor regressor object.
        """
        return ParametricRegressor(
            regressor, features, self.appearance_model, self.transform,
            self.update)


class SemiParametricClassifierBasedRegressorTrainer(
        NonParametricRegressorTrainer):
    r"""
    Class for training a Non-Parametric Classifier-Based Regressor.

    Parameters
    ----------
    classifiers: list of `menpo.fitmultilevel.clm.classifierfunctions`
        List of classifiers.
    transform: `model.transform.affine`
        The transform used for warping.
    reference_shape: PointCloud
        The reference shape that will be used.
    regression_type: function/closure, Optional
        A function/closure that defines the regression technique to be used.
        Examples of such closures can be found in
        `menpo.fit.regression.regressionfunctions.py`

        Default: mlr
    patch_shape: tuple, Optional
        The shape of the patches that will be extracted.

        Default: (16, 16)
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

        Default: False
    n_perturbations: int, Optional
        Defines the number of perturbations that will be applied to the
        training shapes.

        Default: 10
    interpolator: string
        Specifies the interpolator used in warping.

        Default: 'scipy'
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
        r"""
        Method that extracts the features for the regression, which in this
        case are patch based.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image.
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        """
        patches = extract_local_patches(image, shape, self.sampling_grid)
        features = [clf(np.reshape(p.pixels, (-1, p.n_channels)))
                    for (clf, p) in zip(self.classifiers, patches)]
        return np.hstack((np.asarray(features).ravel(), 1))

    def _create_fitting(self, image, shapes, gt_shape=None):
        r"""
        Method that creates the fitting result object.

        Parameters
        ----------
        image: `menpo.image.MaskedImage`
            The image object.
        shapes: `menpo.shape.PointCloud` list
            The shapes.
        gt_shape: `menpo.shape.PointCloud`
            The ground truth shape.
        """
        return SemiParametricFittingResult(image, self, parameters=[shapes],
                                           gt_shape=gt_shape)

    def delta_ps(self, gt_shape, perturbed_shape):
        r"""
        Method to generate the delta_ps for the regression.

        Parameters
        ----------
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        perturbed_shape: :class:`menpo.shape.PointCloud`
            The perturbed shape.
        """
        self.transform.set_target(gt_shape)
        gt_ps = self.transform.as_vector()
        self.transform.set_target(perturbed_shape)
        perturbed_ps = self.transform.as_vector()
        return gt_ps - perturbed_ps

    def _build_regressor(self, regressor, features, ):
        r"""
        Method to build the SemiParametricRegressor regressor object.
        """
        return SemiParametricRegressor(regressor, features, self.transform,
                                       self.update)
