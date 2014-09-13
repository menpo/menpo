from __future__ import division, print_function
import abc
import numpy as np

from menpo.transform import Scale, AlignmentSimilarity
from menpo.model.modelinstance import PDM, OrthoPDM
from menpo.transform.modeldriven import ModelDrivenTransform, OrthoMDTransform
from menpo.visualize import print_dynamic, progress_bar_str

from menpo.fit.regression.trainer import (
    NonParametricRegressorTrainer, ParametricRegressorTrainer,
    SemiParametricClassifierBasedRegressorTrainer)
from menpo.fit.regression.regressionfunctions import mlr
from menpo.fit.regression.parametricfeatures import weights
from menpo.shape import mean_pointcloud
from menpo.fitmultilevel import checks
from menpo.feature import sparse_hog, no_op

from .fitter import SDMFitter, SDAAMFitter, SDCLMFitter


def check_regression_features(regression_features, n_levels):
    try:
        return checks.check_list_callables(regression_features, n_levels)
    except ValueError:
        raise ValueError("regression_features must be a callable or a list of "
                         "{} callables".format(n_levels))


def check_regression_type(regression_type, n_levels):
    r"""
    Checks the regression type (method) per level.

    It must be a callable or a list of those from the family of
    functions defined in :ref:`regression_functions`

    Parameters
    ----------
    regression_type : `function` or list of those
        The regression type to check.

    n_levels : `int`
        The number of pyramid levels.

    Returns
    -------
    regression_type_list : `list`
        A list of regression types that has length ``n_levels``.
    """
    try:
        return checks.check_list_callables(regression_type, n_levels)
    except ValueError:
        raise ValueError("regression_type must be a callable or a list of "
                         "{} callables".format(n_levels))


def check_n_permutations(n_permutations):
    if n_permutations < 1:
        raise ValueError("n_permutations must be > 0")


class SDTrainer(object):
    r"""
    Mixin for Supervised Descent Trainers.

    Parameters
    ----------
    regression_type : `function`, or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified rergession
        type will be applied to all pyramid levels.

        Examples of such closures can be found in :ref:`regression_functions`.

    regression_features :`` None`` or `callable` or `[callable]`, optional
        The features that are used during the regression.

        If `list`, a regression feature is defined per level.

        If not list or list with length ``1``, the specified regression feature
        will be used for all levels.

        Depending on the :map:`SDTrainer` object, this parameter can take
        different types.

    features : `callable` or ``[callable]``, optional
        Defines the features that will be extracted from the image.
        If list of length ``n_levels``, then a feature is defined per level.
        However, this requires that the ``pyramid_on_features`` flag is
        ``False``, so that the features are extracted at each level.
        The first element of the list specifies the features to be extracted
        at the lowest pyramidal level and so on.

        If not a list:
            If ``pyramid_on_features`` is ``True``, the specified feature will
            be applied to the highest level.

            If ``pyramid_on_features`` is ``False``, the specified feature will
            be applied to all pyramid levels.

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    pyramid_on_features : `boolean`, optional
        If ``True``, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.

        If ``False``, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.
`
    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.

    Returns
    -------
    fitter : :map:`MultilevelFitter`
        The fitter object.

    Raises
    ------
    ValueError
        ``regression_type`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    ValueError
        n_levels must be `int` > ``0``
    ValueError
        ``downscale`` must be >= ``1``
    ValueError
        ``n_perturbations`` must be > 0
    ValueError
        ``features`` must be a `string` or a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, regression_type=mlr, regression_features=None,
                 features=no_op, n_levels=3, downscale=1.2,
                 pyramid_on_features=True, noise_std=0.04, rotation=False,
                 n_perturbations=10):

        # general deformable model checks
        checks.check_n_levels(n_levels)
        checks.check_downscale(downscale)
        features = checks.check_features(features, n_levels,
                                         pyramid_on_features)

        # SDM specific checks
        regression_type_list = check_regression_type(regression_type,
                                                     n_levels)
        regression_features = check_regression_features(regression_features,
                                                        n_levels)
        check_n_permutations(n_perturbations)

        # store parameters
        self.regression_type = regression_type_list
        self.regression_features = regression_features
        self.features = features
        self.n_levels = n_levels
        self.downscale = downscale
        self.pyramid_on_features = pyramid_on_features
        self.noise_std = noise_std
        self.rotation = rotation
        self.n_perturbations = n_perturbations

    def train(self, images, group=None, label=None, verbose=False, **kwargs):
        r"""
        Trains a Supervised Descent Regressor given a list of landmarked
        images.

        Parameters
        ----------
        images: list of :map:`MaskedImage`
            The set of landmarked images from which to build the SD.
        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.
        label: `string`, optional
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        verbose: `boolean`, optional
            Flag that controls information and progress printing.
        """
        if verbose:
            print_dynamic('- Computing reference shape')
        self.reference_shape = self._compute_reference_shape(images, group,
                                                             label)
        # store number of training images
        self.n_training_images = len(images)

        # normalize the scaling of all images wrt the reference_shape size
        self._rescale_reference_shape()
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, verbose=verbose)

        # create pyramid
        generators = self._create_pyramid(normalized_images, self.n_levels,
                                          self.downscale,
                                          self.pyramid_on_features,
                                          self.features, verbose=verbose)

        # get feature images of all levels
        images = self._apply_pyramid_on_images(
            generators, self.n_levels, self.pyramid_on_features,
            self.features, verbose=verbose)

        # this .reverse sets the lowest resolution as the first level
        images.reverse()

        # extract the ground truth shapes
        gt_shapes = [[i.landmarks[group][label] for i in img]
                     for img in images]

        # build the regressors
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building regressors for each of the {} '
                              'pyramid levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building regressors\n')

        regressors = []
        # for each pyramid level (low --> high)
        for j, (level_images, level_gt_shapes) in enumerate(zip(images,
                                                                gt_shapes)):
            if verbose:
                if self.n_levels == 1:
                    print_dynamic('\n')
                elif self.n_levels > 1:
                    print_dynamic('\nLevel {}:\n'.format(j + 1))

            # build regressor
            trainer = self._set_regressor_trainer(j)
            if j == 0:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          verbose=verbose, **kwargs)
            else:
                regressor = trainer.train(level_images, level_gt_shapes,
                                          level_shapes, verbose=verbose,
                                          **kwargs)

            if verbose:
                print_dynamic('- Perturbing shapes...')
            level_shapes = trainer.perturb_shapes(gt_shapes[0])

            regressors.append(regressor)
            count = 0
            total = len(regressors) * len(images[0]) * len(level_shapes[0])
            for k, r in enumerate(regressors):

                test_images = images[k]
                test_gt_shapes = gt_shapes[k]

                fitting_results = []
                for (i, gt_s, level_s) in zip(test_images, test_gt_shapes,
                                              level_shapes):
                    fr_list = []
                    for ls in level_s:
                        parameters = r.get_parameters(ls)
                        fr = r.fit(i, parameters)
                        fr.gt_shape = gt_s
                        fr_list.append(fr)
                        count += 1

                    fitting_results.append(fr_list)
                    if verbose:
                        print_dynamic('- Fitting shapes: {}'.format(
                            progress_bar_str((count + 1.) / total,
                                             show_bar=False)))

                level_shapes = [[Scale(self.downscale,
                                       n_dims=self.reference_shape.n_dims
                                       ).apply(fr.final_shape)
                                 for fr in fr_list]
                                for fr_list in fitting_results]

            if verbose:
                print_dynamic('- Fitting shapes: computing mean error...')
            mean_error = np.mean(np.array([fr.final_error()
                                           for fr_list in fitting_results
                                           for fr in fr_list]))
            if verbose:
                print_dynamic("- Fitting shapes: mean error "
                              "is {0:.6f}.\n".format(mean_error))

        return self._build_supervised_descent_fitter(regressors)

    @classmethod
    def _normalization_wrt_reference_shape(cls, images, group, label,
                                           reference_shape, verbose=False):
        r"""
        Normalizes the images sizes with respect to the reference
        shape (mean shape) scaling. This step is essential before building a
        deformable model.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the model.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        reference_shape : :map:`PointCloud`
            The reference shape that is used to resize all training images to
            a consistent object size.

        verbose: bool, optional
            Flag that controls information and progress printing.

        Returns
        -------
        normalized_images : :map:`MaskedImage` list
            A list with the normalized images.
        """
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images),
                                     show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                reference_shape, group=group, label=label))

        if verbose:
            print_dynamic('- Normalizing images size: Done\n')
        return normalized_images

    @classmethod
    def _create_pyramid(cls, images, n_levels, downscale, pyramid_on_features,
                        features, verbose=False):
        r"""
        Function that creates a generator function for Gaussian pyramid. The
        pyramid can be created either on the feature space or the original
        (intensities) space.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        n_levels : `int`
            The number of multi-resolution pyramidal levels to be used.

        downscale : `float`
            The downscale factor that will be used to create the different
            pyramidal levels.

        pyramid_on_features : `boolean`
            If ``True``, the features are extracted at the highest level and the
            pyramid is created on the feature images.

            If ``False``, the pyramid is created on the original (intensities)
            space.

        features : list with `string` or `function` or ``None``
            In case ``pyramid_on_features`` is ``True``, ``features[0]``
            will be used as features type.

        verbose : `boolean`, Optional
            Flag that controls information and progress printing.

        Returns
        -------
        generator : `function`
            The generator function of the Gaussian pyramid.
        """
        if pyramid_on_features:
            # compute features at highest level
            feature_images = []
            for c, i in enumerate(images):
                if verbose:
                    print_dynamic('- Computing feature space: {}'.format(
                        progress_bar_str((c + 1.) / len(images),
                                         show_bar=False)))
                feature_images.append(features[0](i))
            if verbose:
                print_dynamic('- Computing feature space: Done\n')

            # create pyramid on feature_images
            generator = [i.gaussian_pyramid(n_levels=n_levels,
                                            downscale=downscale)
                         for i in feature_images]
        else:
            # create pyramid on intensities images
            # features will be computed per level
            generator = [i.gaussian_pyramid(n_levels=n_levels,
                                            downscale=downscale)
                         for i in images]
        return generator

    @classmethod
    def _apply_pyramid_on_images(cls, generators, n_levels,
                                 pyramid_on_features, features,
                                 verbose=False):
        r"""
        Function that applies the generators of a pyramid on images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        generators : list of generator `function`
            The generator functions of the Gaussian pyramid for all images.

        n_levels: `int`
            The number of multi-resolution pyramidal levels to be used.

        pyramid_on_features: boolean
            If ``True``, the features are extracted at the highest level and the
            pyramid is created on the feature images.
            If ``False``, the pyramid is created on the original (intensities)
            space.

        features: list of length ``n_levels`` with `string` or `function` or ``None``
            The feature type per level to be used in case
            ``pyramid_on_features`` is enabled.

        verbose: `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        feature_images: list of lists of :map:`MaskedImage`
            The set of pyramidal images.
        """
        feature_images = []
        for j in range(n_levels):
            # since generators are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = n_levels - j - 1

            if verbose:
                level_str = '- Apply pyramid: '
                if n_levels > 1:
                    level_str = '- Apply pyramid: [Level {} - '.format(j + 1)

            current_images = []
            if pyramid_on_features:
                # features are already computed, so just call generator
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Rescaling feature space - {}]'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    current_images.append(next(g))
            else:
                # extract features of images returned from generator
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Computing feature space - {}]'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    current_images.append(features[rj](next(g)))
            feature_images.append(current_images)
        if verbose:
            print_dynamic('- Apply pyramid: Done\n')
        return feature_images

    @abc.abstractmethod
    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on the given images shapes.
        """
        pass

    def _rescale_reference_shape(self):
        r"""
        Function that rescales the reference shape w.r.t. to
        ``normalization_diagonal`` parameter.
        """
        pass

    @abc.abstractmethod
    def _set_regressor_trainer(self, **kwargs):
        r"""
        Function that sets the regression object to be one from
        :map:`RegressorTrainer`,
        """
        pass

    @abc.abstractmethod
    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object.

        Parameters
        ----------
        regressors : list of :map:`RegressorTrainer`
            The list of regressors.

        Returns
        -------
        fitter : :map:`SDMFitter`
            The SDM fitter object.
        """
        pass


class SDMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Method using Non-Parametric
    Regression.

    Parameters
    ----------
    regression_type : `function` or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        The function/closures should be one of the methods defined in
        :ref:`regression_functions`

    regression_features: ``None`` or  `callable` or `[callable]`, optional
        If list of length ``n_levels``, then a feature is defined per level.

        If not a list, then the specified feature will be applied to all
        pyramid levels.

        Per level:
            If ``None``, no features are extracted, thus specified
            ``features`` is used in the regressor.

            It is recommended to set the desired features using this option,
            leaving ``features`` equal to :map:`no_op`. This means that the
            images will remain in the intensities space and the features will
            be extracted by the regressor.

    patch_shape: tuple of `int`
        The shape of the patches used by the SDM.

    features : `callable` or `[callable]`, optional
        If list of length ``n_levels``, then a feature is defined per level.
        However, this requires that the ``pyramid_on_features`` flag is
        ``False``, so that the features are extracted at each level.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If not a list:
            If ``pyramid_on_features`` is ``True``, the specified feature will
            be applied to the highest level.

            If ``pyramid_on_features`` is ``False``, the specified feature will
            be applied to all pyramid levels.

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    pyramid_on_features : `boolean`, optional
        If ``True``, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.

        If ``False``, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        initial shape.

    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the initial shape.

    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the shapes.

    normalization_diagonal : `int` >= ``20``, optional
        During training, all images are rescaled to ensure that the scale of
        their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.

        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    Raises
    ------
    ValueError
        ``regression_features`` must be ``None`` or a `string` or a `function`
        or a list of those containing 1 or ``n_level`` elements
    """
    def __init__(self, regression_type=mlr, regression_features=sparse_hog,
                 patch_shape=(16, 16), features=no_op, n_levels=3,
                 downscale=1.5, pyramid_on_features=False, noise_std=0.04,
                 rotation=False, n_perturbations=10,
                 normalization_diagonal=None):
        # in the SDM context regression features are image features,
        # so check them
        regression_features = checks.check_features(regression_features,
                                                    n_levels,
                                                    pyramid_on_features)
        super(SDMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            features=features, n_levels=n_levels, downscale=downscale,
            pyramid_on_features=pyramid_on_features, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal
        self.pyramid_on_features = pyramid_on_features

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on the given images.
        """
        shapes = [i.landmarks[group][label] for i in images]
        return mean_pointcloud(shapes)

    def _rescale_reference_shape(self):
        r"""
        Function that rescales the reference shape w.r.t. to
        ``normalization_diagonal`` parameter.
        """
        if self.normalization_diagonal:
            x, y = self.reference_shape.range()
            scale = self.normalization_diagonal / np.sqrt(x**2 + y**2)
            Scale(scale, self.reference_shape.n_dims).apply_inplace(
                self.reference_shape)

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`NonParametricRegressorTrainer`.

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer : :map:`NonParametricRegressorTrainer`
            The regressor object.
        """
        return NonParametricRegressorTrainer(
            self.reference_shape, regression_type=self.regression_type[level],
            regression_features=self.regression_features[level],
            patch_shape=self.patch_shape, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object.

        Parameters
        ----------
        regressors : list of :map:`RegressorTrainer`
            The list of regressors.

        Returns
        -------
        fitter : :map:`SDMFitter`
            The SDM fitter object.
        """
        return SDMFitter(regressors, self.n_training_images, self.features,
                         self.reference_shape, self.downscale,
                         self.pyramid_on_features)


class SDAAMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Regressor for a given Active
    Appearance Model, thus uses Parametric Regression.

    Parameters
    ----------
    aam : :map:`AAM`
        The trained AAM object.

    regression_type: `function` or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        The function/closures should be one of the methods defined in
        :ref:`regression_functions`

    regression_features: `function` or list of those, optional
        If list of length ``n_levels``, then a feature is defined per level.

        If not a list or a list with length ``1``, then the specified feature
        will be applied to all pyramid levels.

        The function/closures should be one of the methods defined in
        :ref:`parametric_features`.

    noise_std : `float`, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.

    update : {'additive', 'compositional'}
        Defines the way that the warp will be updated.

    md_transform: :map:`ModelDrivenTransform`, optional
        The model driven transform class to be used.

    global_transform : :map:`Affine`, optional
        The global transform class to be used by the previous
        ``md_transform_cls``. Currently, only
        :map:`AlignmentSimilarity` is supported.

    n_shape : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of shape components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available shape components
            (``n_active_components``)will be used.

            If `int` > ``1``, a specific number of shape components is
            specified.

            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.

    n_appearance : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of appearance components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available appearance components
            (``n_active_components``) will be used.
            
            If `int > ``1``, a specific number of appearance components is
            specified.
            
            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.

    Raises
    -------
    ValueError
        n_shape can be an integer or a float or None or a list containing 1
        or ``n_levels`` of those
    ValueError
        n_appearance can be an integer or a float or None or a list containing
        1 or ``n_levels`` of those
    ValueError
        ``regression_features`` must be a `function` or a list of those
        containing ``1`` or ``n_levels`` elements
    """
    def __init__(self, aam, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                 update='compositional', md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None):
        super(SDAAMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            features=aam.features, n_levels=aam.n_levels,
            downscale=aam.downscale,
            pyramid_on_features=aam.pyramid_on_features, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.aam = aam
        self.update = update
        self.md_transform = md_transform
        self.global_transform = global_transform

        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.aam.n_levels:
                for sm, n in zip(self.aam.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.aam.n_levels))

        # check n_appearance parameter
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.aam.n_levels))

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that returns the reference shape computed during AAM building.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape computed based on.
        """
        return self.aam.reference_shape

    def _normalize_object_size(self, images, group, label):
        r"""
        Function that normalizes the images sizes with respect to the reference
        shape (mean shape) scaling.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the model.

        group : `string`
            The key of the landmark set that should be used. If ```None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        normalized_images : :map:`MaskedImage` list
            A list with the normalized images.
        """
        return [i.rescale_to_reference_shape(self.reference_shape,
                                             group=group, label=label)
                for i in images]

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`ParametricRegressorTrainer`.

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer: :map:`ParametricRegressorTrainer`
            The regressor object.
        """
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
            regression_type=self.regression_type[level],
            regression_features=self.regression_features[level],
            update=self.update, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object for AAMs.

        Parameters
        ----------
        regressors : :map:`RegressorTrainer`
            The regressor to build with.

        Returns
        -------
        fitter : :map:`SDAAMFitter`
            The SDM fitter object.
        """
        return SDAAMFitter(self.aam, regressors, self.n_training_images)


class SDCLMTrainer(SDTrainer):
    r"""
    Class that trains Supervised Descent Regressor for a given Constrained
    Local Model, thus uses Semi Parametric Classifier-Based Regression.

    Parameters
    ----------
    clm : :map:`CLM`
        The trained CLM object.

    regression_type: `function` or list of those, optional
        If list of length ``n_levels``, then a regression type is defined per
        level.

        If not a list or a list with length ``1``, then the specified regression
        type will be applied to all pyramid levels.

        The function/closures should be one of the methods defined in
        :ref:`regression_functions`.

    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        training shapes.

    rotation : `boolean`, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the training shapes.

    n_perturbations : `int` > ``0``, optional
        Defines the number of perturbations that will be applied to the
        training shapes.

    pdm_transform : :map:`ModelDrivenTransform`, optional
        The point distribution transform class to be used.

    global_transform : :map:`Affine`, optional
        The global transform class to be used by the previous
        ``md_transform_cls``. Currently, only
        :map:`AlignmentSimilarity` is supported.

    n_shape : `int` > ``1`` or ``0`` <= `float` <= ``1`` or ``None``, or a list of those, optional
        The number of shape components to be used per fitting level.

        If list of length ``n_levels``, then a number of components is defined
        per level. The first element of the list corresponds to the lowest
        pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        components will be used for all levels.

        Per level:
            If ``None``, all the available shape components
            (``n_active_components``) will be used.

            If `int` > ``1``, a specific number of shape components is
            specified.
            
            If ``0`` <= `float` <= ``1``, it specifies the variance percentage
            that is captured by the components.

    Raises
    -------
    ValueError
        ``n_shape`` can be an integer or a `float` or ``None`` or a list
        containing ``1`` or ``n_levels`` of those.
    """
    def __init__(self, clm, regression_type=mlr, noise_std=0.04,
                 rotation=False, n_perturbations=10, pdm_transform=OrthoPDM,
                 global_transform=AlignmentSimilarity, n_shape=None):
        super(SDCLMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=[None] * clm.n_levels,
            features=clm.features, n_levels=clm.n_levels,
            downscale=clm.downscale,
            pyramid_on_features=clm.pyramid_on_features, noise_std=noise_std,
            rotation=rotation, n_perturbations=n_perturbations)
        self.clm = clm
        self.patch_shape = clm.patch_shape
        self.pdm_transform = pdm_transform
        self.global_transform = global_transform

        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.clm.n_levels > 1:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.clm.n_levels:
                for sm, n in zip(self.clm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.clm.n_levels))

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that returns the reference shape computed during CLM building.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images.

        group : `string`
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`
            The label of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape : :map:`PointCloud`
            The reference shape.
        """
        return self.clm.reference_shape

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        :map:`SemiParametricClassifierBasedRegressorTrainer`

        Parameters
        ----------
        level : `int`
            The scale level.

        Returns
        -------
        trainer: :map:`SemiParametricClassifierBasedRegressorTrainer`
            The regressor object.
        """
        clfs = self.clm.classifiers[level]
        sm = self.clm.shape_models[level]

        if self.pdm_transform is not PDM:
            pdm_transform = self.pdm_transform(sm, self.global_transform)
        else:
            pdm_transform = self.pdm_transform(sm)

        return SemiParametricClassifierBasedRegressorTrainer(
            clfs, pdm_transform, self.reference_shape,
            regression_type=self.regression_type[level],
            patch_shape=self.patch_shape, update='additive',
            noise_std=self.noise_std, rotation=self.rotation,
            n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object for CLMs.

        Parameters
        ----------
        regressors : :map:`RegressorTrainer`
            Regressor to train with.

        Returns
        -------
        fitter : :map:`SDCLMFitter`
            The SDM fitter object.
        """
        return SDCLMFitter(self.clm, regressors, self.n_training_images)
