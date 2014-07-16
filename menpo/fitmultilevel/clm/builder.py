from __future__ import division, print_function
import numpy as np

from menpo.image import Image
from menpo.fitmultilevel.builder import DeformableModelBuilder
from menpo.fitmultilevel.functions import build_sampling_grid
from menpo.fitmultilevel.featurefunctions import compute_features, sparse_hog
from menpo.visualize import print_dynamic, progress_bar_str

from .classifierfunctions import classifier, linear_svm_lr


class CLMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Constrained Local Models.

    Parameters
    ----------
    classifier_type : ``classifier_closure`` or list of those
        If list of length ``n_levels``, then a classifier function is defined
        per level. The first element of the list specifies the classifier to be
        used at the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified classifier
        function will be used for all levels.

        Per level:
             A closure implementing a binary classifier.

        Examples of such closures can be found in
        :ref:`clm_builders`

    patch_shape : tuple of `int`
        The shape of the patches used by the previous classifier closure.

    feature_type : ``None`` or `string` or `function` or list of those, optional
        If list of length ``n_levels``, then a feature is defined per level.
        However, this requires that the ``pyramid_on_features`` flag is
        disabled, so that the features are extracted at each level.
        The first element of the list specifies the features to be extracted
        at the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then:
            If ``pyramid_on_features`` is ``True``, the specified feature will
            be applied to the highest level.
            If ``pyramid_on_features`` is ``False``, the specified feature will
            be applied to all pyramid levels.

        Per level:
            If ``None``, the appearance model will be built using the original
            image representation, i.e. no features will be extracted from the
            original images.

        If `string`, image features will be computed by executing::

           feature_image = getattr(image.features, feature_type[level])()

        for each pyramidal level. For this to work properly each `string`
        needs to be one of Menpo's standard image feature methods
        ('igo', 'hog', ...).
        Note that, in this case, the feature computation will be
        carried out using the default options.

        Non-default feature options and new experimental features can be
        defined using functions. In this case, the functions must
        receive an image as input and return a particular feature
        representation of that image. For example::

            def igo_double_from_std_normalized_intensities(image)
                image = deepcopy(image)
                image.normalize_std_inplace()
                return image.feature_type.igo(double_angles=True)

        See :map:`ImageFeatures` for details more details on
        Menpo's standard image features and feature options.

    normalization_diagonal : `int` >= ``20``, optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If `int`, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the ``normalization_diagonal``
        value.
        If ``None``, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

    n_levels : `int` > ``0``, optional
        The number of multi-resolution pyramidal levels to be used.

    downscale : `float` >= ``1``, optional
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames will be the mean shapes of each
        pyramid level, so the shape models will be scaled.

        If ``False``, the reference frames of all levels will be the mean shape
        of the highest level, so the shape models will not be scaled; they will
        have the same size.

    pyramid_on_features : `boolean`, optional
        If ``True``, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.

        If ``False``, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

    max_shape_components : ``None`` or `int` > ``0`` or ``0`` <= `float` <= ``1`` or list of those, optional
        If list of length ``n_levels``, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length ``1``, then the specified number of
        shape components will be used for all levels.

        Per level:
            If `int`, it specifies the exact number of components to be
            retained.

            If `float`, it specifies the percentage of variance to be retained.

            If ``None``, all the available components are kept
            (100% of variance).

    boundary : `int` >= ``0``, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

    interpolator : `string`, optional
        The interpolator that should be used to perform the warps.

    Returns
    -------
    clm : :map:`CLMBuilder`
        The CLM Builder object
    """
    def __init__(self, classifier_type=linear_svm_lr, patch_shape=(5, 5),
                 feature_type=sparse_hog, normalization_diagonal=None,
                 n_levels=3, downscale=1.1, scaled_shape_models=True,
                 pyramid_on_features=False, max_shape_components=None,
                 boundary=3, interpolator='scipy'):
        # check parameters
        self.check_n_levels(n_levels)
        self.check_downscale(downscale)
        self.check_normalization_diagonal(normalization_diagonal)
        self.check_boundary(boundary)
        max_shape_components = self.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        feature_type = self.check_feature_type(feature_type, n_levels,
                                               pyramid_on_features)
        classifier_type = check_classifier_type(classifier_type, n_levels)
        patch_shape = check_patch_shape(patch_shape)

        # store parameters
        self.classifier_type = classifier_type
        self.patch_shape = patch_shape
        self.feature_type = feature_type
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.max_shape_components = max_shape_components
        self.boundary = boundary
        self.interpolator = interpolator

    def build(self, images, group=None, label='all', verbose=False):
        r"""
        Builds a Multilevel Constrained Local Model from a list of
        landmarked images.

        Parameters
        ----------
        images : list of :map:`Image`
            The set of landmarked images from which to build the AAM.

        group : string, Optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`, optional
            The label of of the landmark manager that you wish to use. If
            ``None``, the convex hull of all landmarks is used.

        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        clm : :map:`CLM`
            The CLM object
        """
        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            self._normalization_wrt_reference_shape(
                images, group, label, self.normalization_diagonal,
                self.interpolator, verbose=verbose)

        # create pyramid
        generators = self._create_pyramid(normalized_images, self.n_levels,
                                          self.downscale,
                                          self.pyramid_on_features,
                                          self.feature_type, verbose=verbose)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        classifiers = []
        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters of type list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get images of current level
            feature_images = []
            if self.pyramid_on_features:
                # features are already computed, so just call generator
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Rescaling feature space - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    feature_images.append(g.next())
            else:
                # extract features of images returned from generator
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Computing feature space - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    feature_images.append(compute_features(
                        g.next(), self.feature_type[rj]))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label].lms for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = self._build_shape_model(
                train_shapes, self.max_shape_components[rj])

            # add shape model to the list
            shape_models.append(shape_model)

            # build classifiers
            sampling_grid = build_sampling_grid(self.patch_shape)
            n_points = shapes[0].n_points
            level_classifiers = []
            for k in range(n_points):
                if verbose:
                    print_dynamic('{}Building classifiers - {}'.format(
                        level_str,
                        progress_bar_str((k + 1.) / n_points,
                                         show_bar=False)))

                positive_labels = []
                negative_labels = []
                positive_samples = []
                negative_samples = []

                for i, s in zip(feature_images, shapes):

                    max_x = i.shape[0] - 1
                    max_y = i.shape[1] - 1

                    point = (np.round(s.points[k, :])).astype(int)
                    patch_grid = sampling_grid + point[None, None, ...]
                    positive, negative = get_pos_neg_grid_positions(
                        patch_grid, positive_grid_size=(1, 1))

                    x = positive[:, 0]
                    y = positive[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    positive_sample = i.pixels[positive[:, 0],
                                               positive[:, 1], :]
                    positive_samples.append(positive_sample)
                    positive_labels.append(np.ones(positive_sample.shape[0]))

                    x = negative[:, 0]
                    y = negative[:, 1]
                    x[x > max_x] = max_x
                    y[y > max_y] = max_y
                    x[x < 0] = 0
                    y[y < 0] = 0

                    negative_sample = i.pixels[x, y, :]
                    negative_samples.append(negative_sample)
                    negative_labels.append(-np.ones(negative_sample.shape[0]))

                positive_samples = np.asanyarray(positive_samples)
                positive_samples = np.reshape(positive_samples,
                                              (-1, positive_samples.shape[-1]))
                positive_labels = np.asanyarray(positive_labels).flatten()

                negative_samples = np.asanyarray(negative_samples)
                negative_samples = np.reshape(negative_samples,
                                              (-1, negative_samples.shape[-1]))
                negative_labels = np.asanyarray(negative_labels).flatten()

                X = np.vstack((positive_samples, negative_samples))
                t = np.hstack((positive_labels, negative_labels))

                clf = classifier(X, t, self.classifier_type[rj])
                level_classifiers.append(clf)

            # add level classifiers to the list
            classifiers.append(level_classifiers)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        classifiers.reverse()
        n_training_images = len(images)

        return CLM(shape_models, classifiers, n_training_images,
                   self.patch_shape, self.feature_type, self.reference_shape,
                   self.downscale, self.scaled_shape_models,
                   self.pyramid_on_features, self.interpolator)


class CLM(object):
    r"""
    Constrained Local Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the CLM.

    classifiers : ``classifier_closure`` list of lists
        A list containing the list of classifier_closures per each pyramidal
        level of the CLM.

    n_training_images : `int`
        The number of training images used to build the AAM.

    patch_shape : tuple of `int`
        The shape of the patches used to train the classifiers.

    feature_type : ``None`` or `string` or `function` or list of those
        The image feature that was be used to build the ``appearance_models``.
        Will subsequently be used by fitter objects using this class to fit to
        novel images.

        If list of length ``n_levels``, then a feature was defined per level.
        This means that the ``pyramid_on_features`` flag was ``False``
        and the features were extracted at each level. The first element of
        the list specifies the features of the lowest pyramidal level and so
        on.

        If not a list or a list with length ``1``, then:
            If ``pyramid_on_features`` is ``True``, the specified feature
            was applied to the highest level.

            If ``pyramid_on_features`` is ``False``, the specified feature was
            applied to all pyramid levels.

        Per level:
            If ``None``, the appearance model was built using the original
            image representation, i.e. no features will be extracted from the
            original images.

            If `string`, the appearance model was built using one of Menpo's
            default built-in feature representations - those
            accessible at ``image.features.some_feature()``. Note that this case
            can only be used with default feature parameters - for custom
            feature weights, use the functional form of this argument instead.

            If `function`, the user can directly provide the feature that was
            calculated on the images. This class will simply invoke this
            function, passing in as the sole argument the image to be fitted,
            and expect as a return type an :map:`Image` representing the feature
            calculation ready for further fitting. See the examples for
            details.

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, Optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

    pyramid_on_features : `boolean`, optional
        If True, the feature space was computed once at the highest scale and
        the Gaussian pyramid was applied on the feature images.
        If False, the Gaussian pyramid was applied on the original images
        (intensities) and then features were extracted at each level.

    interpolator : `string`
        The interpolator that was used to build the CLM.
    """
    def __init__(self, shape_models, classifiers, n_training_images,
                 patch_shape, feature_type, reference_shape, downscale,
                 scaled_shape_models, pyramid_on_features, interpolator):
        self.shape_models = shape_models
        self.classifiers = classifiers
        self.n_training_images = n_training_images
        self.patch_shape = patch_shape
        self.feature_type = feature_type
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.interpolator = interpolator

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the CLM.

        :type: `int`
        """
        return len(self.shape_models)

    @property
    def n_classifiers_per_level(self):
        """
        The number of classifiers per pyramidal level of the CLM.

        :type: `int`
        """
        return [len(clf) for clf in self.classifiers]

    def instance(self, shape_weights=None, level=-1):
        r"""
        Generates a novel CLM instance given a set of shape weights. If no
        weights are provided, the mean CLM instance is returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If `None`, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        shape_instance : :map:`PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def random_instance(self, level=-1):
        r"""
        Generates a novel random CLM instance.

        Parameters
        -----------
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        shape_instance : :map:`PointCloud`
            The novel CLM instance.
        """
        sm = self.shape_models[level]
        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        return shape_instance

    def response_image(self, image, group=None, label='all', level=-1):
        r"""
        Generates a response image result of applying the classifiers of a
        particular pyramidal level of the CLM to an image.

        Parameters
        -----------
        image: :map:`Image`
            The image.

        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        level: `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
            The response image.
        """
        # rescale image
        image = image.rescale_to_reference_shape(self.reference_shape,
                                                 group=group, label=label)

        # apply pyramid
        if self.n_levels > 1:
            if self.pyramid_on_features:
                # compute features at highest level
                feature_image = compute_features(image, self.feature_type[0])

                # apply pyramid on feature image
                pyramid = feature_image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # get rescaled feature images
                images = list(pyramid)
            else:
                # create pyramid on intensities image
                pyramid = image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)

                # compute features at each level
                images = [compute_features(
                    i, self.feature_type[self.n_levels - j - 1])
                    for j, i in enumerate(pyramid)]
            images.reverse()
        else:
            images = [compute_features(image, self.feature_type[0])]

        # initialize responses
        image = images[level]
        image_pixels = np.reshape(image.pixels, (-1, image.n_channels))
        response_data = np.zeros((image.shape[0], image.shape[1],
                                  self.n_classifiers_per_level[level]))
        # Compute responses
        for j, clf in enumerate(self.classifiers[level]):
            response_data[:, :, j] = np.reshape(clf(image_pixels),
                                                image.shape)
        return Image(image_data=response_data)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        : str
        """
        return 'Constrained Local Model'

    def __str__(self):
        out = "{}\n - {} training images.\n".format(self._str_title,
                                                    self.n_training_images)
        # small strings about number of channels, channels string and downscale
        down_str = []
        for j in range(self.n_levels):
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        temp_img = Image(image_data=np.random.rand(50, 50))
        if self.pyramid_on_features:
            temp = compute_features(temp_img, self.feature_type[0])
            n_channels = [temp.n_channels] * self.n_levels
        else:
            n_channels = []
            for j in range(self.n_levels):
                temp = compute_features(temp_img, self.feature_type[j])
                n_channels.append(temp.n_channels)
        # string about features and channels
        if self.pyramid_on_features:
            if isinstance(self.feature_type[0], str):
                feat_str = "- Feature is {} with ".format(
                    self.feature_type[0])
            elif self.feature_type[0] is None:
                feat_str = "- No features extracted. "
            else:
                feat_str = "- Feature is {} with ".format(
                    self.feature_type[0].func_name)
            if n_channels[0] == 1:
                ch_str = ["channel"]
            else:
                ch_str = ["channels"]
        else:
            feat_str = []
            ch_str = []
            for j in range(self.n_levels):
                if isinstance(self.feature_type[j], str):
                    feat_str.append("- Feature is {} with ".format(
                        self.feature_type[j]))
                elif self.feature_type[j] is None:
                    feat_str.append("- No features extracted. ")
                else:
                    feat_str.append("- Feature is {} with ".format(
                        self.feature_type[j].func_name))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        if self.n_levels > 1:
            if self.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.patch_shape[1], self.patch_shape[0])

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.patch_shape[1], self.patch_shape[0])
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                out = "{0}     - {1} shape components ({2:.2f}% of " \
                      "variance)\n     - {3} {4} classifiers.\n".format(
                      out, self.shape_models[i].n_components,
                      self.shape_models[i].variance_ratio * 100,
                      self.n_classifiers_per_level[i],
                      self.classifiers[i][0].func_name)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - {4} shape components ({5:.2f}% of " \
                  "variance)\n   - {6} {7} classifiers.".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  self.shape_models[0].n_components,
                  self.shape_models[0].variance_ratio * 100,
                  self.n_classifiers_per_level[0],
                  self.classifiers[0][0].func_name)
        return out


def get_pos_neg_grid_positions(sampling_grid, positive_grid_size=(1, 1)):
    r"""
    Divides a sampling grid in positive and negative pixel positions. By
    default only the center of the grid is considered to be positive.
    """
    positive_grid_size = np.array(positive_grid_size)
    mask = np.zeros(sampling_grid.shape[:-1], dtype=np.bool)
    center = np.round(np.array(mask.shape) / 2).astype(int)
    positive_grid_size -= [1, 1]
    start = center - positive_grid_size
    end = center + positive_grid_size + 1
    mask[start[0]:end[0], start[1]:end[1]] = True
    positive = sampling_grid[mask]
    negative = sampling_grid[~mask]
    return positive, negative


def check_classifier_type(classifier_type, n_levels):
    r"""
    Checks the classifier type per level. It must be a classifier
    function closure or a list containing 1 or {n_levels} closures.
    """
    str_error = ("classifier_type must be a classifier function closure "
                 "of a list containing 1 or {} closures").format(n_levels)
    if not isinstance(classifier_type, list):
        classifier_type_list = [classifier_type] * n_levels
    elif len(classifier_type) == 1:
        classifier_type_list = [classifier_type[0]] * n_levels
    elif len(classifier_type) == n_levels:
        classifier_type_list = classifier_type
    else:
        raise ValueError(str_error)
    for clas in classifier_type_list:
        if not hasattr(clas, '__call__'):
            raise ValueError(str_error)
    return classifier_type_list


def check_patch_shape(patch_shape):
    r"""
    Checks the patch shape. It must be a tuple with `int` > ``1``.
    """
    str_error = "patch_size mast be a tuple with two integers"
    if not isinstance(patch_shape, tuple) or len(patch_shape) != 2:
        raise ValueError(str_error)
    for sh in patch_shape:
        if not isinstance(sh, int) or sh < 2:
            raise ValueError(str_error)
    return patch_shape
