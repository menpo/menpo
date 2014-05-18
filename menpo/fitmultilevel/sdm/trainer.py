from __future__ import division, print_function
import abc
import numpy as np

from menpo.transform import Scale, AlignmentSimilarity
from menpo.model.pdm import PDM, OrthoPDM
from menpo.transform.modeldriven import ModelDrivenTransform, OrthoMDTransform
from menpo.visualize import print_dynamic, progress_bar_str

from menpo.fit.regression.trainer import (
    NonParametricRegressorTrainer, ParametricRegressorTrainer,
    SemiParametricClassifierBasedRegressorTrainer)
from menpo.fit.regression.regressionfunctions import mlr
from menpo.fit.regression.parametricfeatures import weights
from menpo.fitmultilevel.functions import mean_pointcloud
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog

from .base import (SDMFitter, SDAAMFitter, SDCLMFitter)


# TODO: document me
class SDTrainer(object):
    r"""
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, regression_type=mlr, regression_features=None,
                 feature_type=None, n_levels=3, downscale=1.2,
                 scaled_shape_models=True, pyramid_on_features=True,
                 noise_std=0.04, rotation=False,
                 n_perturbations=10, interpolator='scipy', **kwargs):

        # check parameters
        self.check_n_levels(n_levels)
        self.check_downscale(downscale)
        feature_type = self.check_feature_type(feature_type, n_levels,
                                               pyramid_on_features)
        regression_features = self.check_feature_type(regression_features,
                                                      n_levels,
                                                      pyramid_on_features)

        # store parameters
        self.regression_type = regression_type
        self.regression_features = regression_features
        self.feature_type = feature_type
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.noise_std = noise_std
        self.rotation = rotation
        self.n_perturbations = n_perturbations
        self.interpolator = interpolator

    def train(self, images, group=None, label='all', verbose=False, **kwargs):
        r"""
        """
        if verbose:
            print_dynamic('- Computing reference shape')
        self.reference_shape = self._compute_reference_shape(images, group,
                                                             label)

        # normalize the scaling of all images wrt the reference_shape size
        self._rescale_reference_shape()
        normalized_images = self._normalization_wrt_reference_shape(
            images, group, label, self.reference_shape, self.interpolator,
            verbose=verbose)

        # create pyramid
        generators = self._create_pyramid(normalized_images, self.n_levels,
                                          self.downscale,
                                          self.pyramid_on_features,
                                          self.feature_type, verbose=verbose)

        # get feature images of all levels
        images = self._apply_pyramid_on_images(
            generators, self.n_levels, self.pyramid_on_features,
            self.feature_type, verbose=verbose)

        # this reverse sets the lowest resolution as the first level
        images.reverse()

        # extract groundtruth shapes
        gt_shapes = [[i.landmarks[group][label].lms for i in img]
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
                print_dynamic('- Generating next level data')
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
                    if verbose:
                        print_dynamic('- Fitting shapes: {}'.format(
                            progress_bar_str((count + 1.) / total,
                                             show_bar=False)))

                level_shapes = [[Scale(self.downscale,
                                       n_dims=self.reference_shape.n_dims
                                       ).apply(f.final_shape)
                                 for f in fitting_sublist]
                                for fitting_sublist in fittings]

            mean_error = np.mean(np.array([f.final_error
                                           for fitting_sublist in fittings
                                           for f in fitting_sublist]))
            if verbose:
                print('- Fitting shapes: Done\n'
                      '  - Mean error is {0:.6f}.'.format(mean_error))

        return self._build_supervised_descent_fitter(regressors)

    @classmethod
    def _normalization_wrt_reference_shape(cls, images, group, label,
                                           reference_shape, interpolator,
                                           verbose=False):
        r"""
        Function that normalizes the images sizes with respect to the reference
        shape (mean shape) scaling. This step is essential before building a
        deformable model.

        The normalization includes:
        1) Computation of the reference shape as the mean shape of the images'
           landmarks.
        2) Scaling of the reference shape using the normalization_diagonal.
        3) Rescaling of all the images so that their shape's scale is in
           correspondence with the reference shape's scale.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images from which to build the model.
        group : string
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.
        label: string
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.
        reference_shape: Pointcloud
            The reference shape that is used to resize all training images to
            a consistent object size.
        interpolator: string
            The interpolator that should be used to perform the warps.
        verbose: bool, Optional
            Flag that controls information and progress printing.

            Default: False

        Returns
        -------
        normalized_images: list of MaskedImage objects
            A list with the normalized images.
        """
        normalized_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images),
                                     show_bar=False)))
            normalized_images.append(i.rescale_to_reference_shape(
                reference_shape, group=group, label=label,
                interpolator=interpolator))

        if verbose:
            print_dynamic('- Normalizing images size: Done\n')
        return normalized_images

    @classmethod
    def _create_pyramid(cls, images, n_levels, downscale, pyramid_on_features,
                        feature_type, verbose=False):
        r"""
        Function that creates a generator function for Gaussian pyramid. The
        pyramid can be created either on the feature space or the original
        (intensities) space.

        Parameters
        ----------
        images: list of :class:`menpo.image.Image`
            The set of landmarked images.
        n_levels: int
            The number of multi-resolution pyramidal levels to be used.
        downscale: float
            The downscale factor that will be used to create the different
            pyramidal levels.
        pyramid_on_features: boolean
            If True, the features are extracted at the highest level and the
            pyramid is created on the feature images.
            If False, the pyramid is created on the original (intensities)
            space.
        feature_type: list of size 1 with str or function/closure or None
            The feature type to be used in case pyramid_on_features is enabled.
        verbose: bool, Optional
            Flag that controls information and progress printing.

            Default: False

        Returns
        -------
        generator: function
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
                feature_images.append(compute_features(i, feature_type[0]))
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
                                 pyramid_on_features, feature_type,
                                 verbose=False):
        r"""
        Function that applies a pyramid genertors on images.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images.
        generators: list of generator functions
            The generator function of the Gaussian pyramid for all images.
        n_levels: int
            The number of multi-resolution pyramidal levels to be used.
        pyramid_on_features: boolean
            If True, the features are extracted at the highest level and the
            pyramid is created on the feature images.
            If False, the pyramid is created on the original (intensities)
            space.
        feature_type: list of size 1 with str or function/closure or None
            The feature type to be used in case pyramid_on_features is enabled.
        verbose: bool, Optional
            Flag that controls information and progress printing.

            Default: False

        Returns
        -------
        feature_images: list of lists of :class:`menpo.image.MaskedImage`
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
                    current_images.append(g.next())
            else:
                # extract features of images returned from generator
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Computing feature space - {}]'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    current_images.append(compute_features(g.next(),
                                                           feature_type[rj]))
            feature_images.append(current_images)
        if verbose:
            print_dynamic('- Apply pyramid: Done\n')
        return feature_images

    #TODO: repeated code from Builder. Should builder and Trainer have a
    # common ancestor???
    @classmethod
    def check_feature_type(cls, feature_type, n_levels, pyramid_on_features):
        r"""
        Checks the feature type per level.
        If pyramid_on_features is False, it must be a string or a
        function/closure or a list of those containing 1 or {n_levels}
        elements.
        If pyramid_on_features is True, it must be a string or a
        function/closure or a list of 1 of those.
        """
        if pyramid_on_features is False:
            feature_type_str_error = ("feature_type must be a str or a "
                                      "function/closure or a list of "
                                      "those containing 1 or {} "
                                      "elements").format(n_levels)
            if not isinstance(feature_type, list):
                feature_type_list = [feature_type] * n_levels
            elif len(feature_type) is 1:
                feature_type_list = [feature_type[0]] * n_levels
            elif len(feature_type) is n_levels:
                feature_type_list = feature_type
            else:
                raise ValueError(feature_type_str_error)
        else:
            feature_type_str_error = ("pyramid_on_features is enabled so "
                                      "feature_type must be a str or a "
                                      "function/closure or a list "
                                      "containing 1 of those")
            if not isinstance(feature_type, list):
                feature_type_list = [feature_type]
            elif len(feature_type) is 1:
                feature_type_list = feature_type
            else:
                raise ValueError(feature_type_str_error)
        for ft in feature_type_list:
            if ft is not None:
                if not isinstance(ft, str):
                    if not hasattr(ft, '__call__'):
                        raise ValueError(feature_type_str_error)
        return feature_type_list

    @classmethod
    def check_n_levels(cls, n_levels):
        r"""
        Checks the number of pyramid levels that must be int > 0.
        """
        if not isinstance(n_levels, int) or n_levels < 1:
            raise ValueError("n_levels must be int > 0")

    @classmethod
    def check_downscale(cls, downscale):
        r"""
        Checks the downscale factor of the pyramid that must be >= 1.
        """
        if downscale < 1:
            raise ValueError("downscale must be >= 1")

    @abc.abstractmethod
    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images.
        group : string
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.
        label: string
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape: Pointcloud
            The reference shape computed based on .
        """
        pass

    def _rescale_reference_shape(self):
        r"""
        Function rescales the reference shape wrt to normalization_diagonal
        parameter.
        """
        pass

    @abc.abstractmethod
    def _set_regressor_trainer(self, **kwargs):
        r"""
        Function that sets the regression object to be one from
        :class: menpo.fit.regression.RegressorTrainer,
        """
        pass

    @abc.abstractmethod
    def _build_supervised_descent_fitter(self, regressors):
        pass


#TODO: Document me
class SDMTrainer(SDTrainer):
    r"""
    Class that trains SDM using Non-Parametric Regression.

    Parameters
    ----------
    regression_type: function/closure
        A function/closure that defines the type of regression.
        Examples of such closures can be found in
        `menpo.fit.regression.trainer`

        Default: mlr
    patch_shape: tuple of ints
        The shape of the patches used by the SDM.

        Default: (16, 16)
    feature_type: None or string or function/closure or list of those, Optional
        If list of length n_levels, then a feature is defined per level.
        However, this requires that the pyramid_on_features flag is disabled,
        so that the features are extracted at each level. The first element of
        the list specifies the features to be extracted at the lowest pyramidal
        level and so on.

        If not a list or a list with length 1, then:
            If pyramid_on_features is True, the specified feature will be
            applied to the highest level.
            If pyramid_on_features is False, the specified feature will be
            applied to all pyramid levels.

        Per level:
        If None, the appearance model will be built using the original image
        representation, i.e. no features will be extracted from the original
        images.

        If string, image features will be computed by executing:

           feature_image = eval('img.feature_type.' +
                                feature_type[level] + '()')

        for each pyramidal level. For this to work properly each string
        needs to be one of menpo's standard image feature methods
        ('igo', 'hog', ...).
        Note that, in this case, the feature computation will be
        carried out using the default options.

        Non-default feature options and new experimental features can be
        defined using functions/closures. In this case, the functions must
        receive an image as input and return a particular feature
        representation of that image. For example:

            def igo_double_from_std_normalized_intensities(image)
                image = deepcopy(image)
                image.normalize_std_inplace()
                return image.feature_type.igo(double_angles=True)

        See `menpo.image.feature.py` for details more details on
        menpo's standard image features and feature options.

        Default: None
    n_levels: int > 0, Optional
        The number of multi-resolution pyramidal levels to be used.

        Default: 3
    downscale: float >= 1, Optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be:
            (downscale ** k) for k in range(n_levels)

        Default: 2
    scaled_shape_models: boolean, Optional
        If True, the reference frames will be the mean shapes of each pyramid
        level, so the shape models will be scaled.
        If False, the reference frames of all levels will be the mean shape of
        the highest level, so the shape models will not be scaled; they will
        have the same size.

        Default: True
    pyramid_on_features: boolean, Optional
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

        Default: True
    noise_std: float, optional
        The standard deviation of the gaussian noise used to produce the
        initial shape.

        Default: 0.04
    rotation: boolean, optional
        Specifies whether ground truth in-plane rotation is to be used
        to produce the initial shape.

        Default: False
    n_perturbations: int > 0, Optional
        Defines the number of perturbations that will be applied to the shapes.

        Default: 10
    normalization_diagonal: int >= 20, Optional
        During training, all images are rescaled to ensure that the scale of
        their landmarks matches the scale of the mean shape.

        If int, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.
        If None, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

        Default: None
    interpolator: string, Optional
        The interpolator in use.

        Default: 'scipy'

    Raises
    -------
    ValueError
        n_levels must be int > 0
    ValueError
        downscale must be >= 1
    ValueError
        normalization_diagonal must be >= 20
    ValueError
        max_shape_components must be None or an int > 0 or a 0 <= float <= 1
        or a list of those containing 1 or {n_levels} elements
    ValueError
        max_appearance_components must be None or an int > 0 or a
        0 <= float <= 1 or a list of those containing 1 or {n_levels} elements
    ValueError
        feature_type must be a str or a function/closure or a list of those
        containing 1 or {n_levels} elements
    ValueError
        pyramid_on_features is enabled so feature_type must be a str or a
        function/closure or a list containing 1 of those
    """
    def __init__(self, regression_type=mlr, regression_features=sparse_hog,
                 patch_shape=(16, 16), feature_type=None, n_levels=3,
                 downscale=1.5, scaled_shape_models=True,
                 pyramid_on_features=False, noise_std=0.04, rotation=False,
                 n_perturbations=10, normalization_diagonal=None,
                 interpolator='scipy'):
        super(SDMTrainer, self).__init__(
            regression_type=regression_type,
            regression_features=regression_features,
            feature_type=feature_type, n_levels=n_levels,
            downscale=downscale, scaled_shape_models=scaled_shape_models,
            pyramid_on_features=pyramid_on_features,
            noise_std=noise_std, rotation=rotation,
            n_perturbations=n_perturbations, interpolator=interpolator)
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal

    def _compute_reference_shape(self, images, group, label):
        r"""
        Function that computes the reference shape, given a set of images.

        Parameters
        ----------
        images: list of :class:`menpo.image.MaskedImage`
            The set of landmarked images.
        group : string
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.
        label: string
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        Returns
        -------
        reference_shape: Pointcloud
            The reference shape computed based on.
        """
        shapes = [i.landmarks[group][label].lms for i in images]
        return mean_pointcloud(shapes)

    def _rescale_reference_shape(self):
        r"""
        Function rescales the reference shape wrt to normalization_diagonal
        parameter.
        """
        if self.normalization_diagonal:
            x, y = self.reference_shape.range()
            scale = self.normalization_diagonal / np.sqrt(x**2 + y**2)
            Scale(scale, self.reference_shape.n_dims).apply_inplace(
                self.reference_shape)

    def _set_regressor_trainer(self, level):
        r"""
        Function that sets the regression class to be the
        NonParametricRegressorTrainer.

        Parameter
        ---------
        level: int
            The scale level.

        Returns
        -------
        trainer: :class: menpo.fit.regression.NonParametricRegressorTrainer
            The regressor object.
        """
        return NonParametricRegressorTrainer(
            self.reference_shape, regression_type=self.regression_type,
            regression_features=self.regression_features[level],
            patch_shape=self.patch_shape, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        r"""
        Builds an SDM fitter object.

        Parameters
        ----------
        regressors: :class: menpo.fit.regression.RegressorTrainer
        """
        return SDMFitter(
            regressors, self.feature_type, self.reference_shape,
            self.downscale, self.scaled_levels, self.interpolator)


#TODO: Document me
class SDAAMTrainer(SDTrainer):
    r"""
    """
    def __init__(self, aam, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                 update='compositional', md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None):
        super(SDAAMTrainer, self).__init__(
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
            regression_type=self.regression_type,
            regression_features=self.regression_features[level],
            update=self.update, noise_std=self.noise_std,
            rotation=self.rotation, n_perturbations=self.n_perturbations,
            interpolator=self.interpolator)

    def _build_supervised_descent_fitter(self, regressors):
        return SDAAMFitter(self.aam, regressors)


#TODO: Document me
#TODO: Finish me
class SDCLMTrainer(SDTrainer):
    r"""
    """
    def __init__(self, clm, regression_type=mlr, regression_features=weights,
                 noise_std=0.04, rotation=False, n_perturbations=10,
                pdm_transform=OrthoPDM,
                global_transform=AlignmentSimilarity, n_shape=None):
        super(SDCLMTrainer, self).__init__(
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
            regression_type=self.regression_type[level],
            patch_shape=self.patch_shape,
            noise_std=self.noise_std, rotation=self.rotation,
            n_perturbations=self.n_perturbations)

    def _build_supervised_descent_fitter(self, regressors):
        return SDCLMFitter(self.clm, regressors)
