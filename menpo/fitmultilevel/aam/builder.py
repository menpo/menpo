from __future__ import division
import numpy as np
from random import sample

from menpo.shape import TriMesh
from menpo.image import MaskedImage
from menpo.transform import Scale, Translation
from menpo.transform.piecewiseaffine import PiecewiseAffine
from menpo.transform.thinplatesplines import ThinPlateSplines
from menpo.model import PCAModel
from menpo.fitmultilevel.builder import DeformableModelBuilder
from menpo.fitmultilevel.featurefunctions import compute_features
from menpo.visualize import print_dynamic, progress_bar_str, print_bytes
from ..functions import mean_pointcloud


class AAMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Active Appearance Models.

    Parameters
    ----------
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

        Default: 'igo'
    transform: :class:`menpo.transform.PureAlignmentTransform`, Optional
        The :class:`menpo.transform.PureAlignmentTransform` that will be
        used to warp the images.

        Default: :class:`menpo.transform.PiecewiseAffine`
    trilist: (t, 3) ndarray, Optional
        Triangle list that will be used to build the reference frame. If None,
        defaults to performing Delaunay triangulation on the points.

        Default: None
    normalization_diagonal: int >= 20, Optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If int, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.
        If None, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

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
        Note that from our experience, if scaled_shape_models is False, AAMs
        tend to have slightly better performance.

        Default: True
    pyramid_on_features: boolean, Optional
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.
        Note that from our experience, if pyramid_on_features is True, AAMs
        tend to have slightly better performance.

        Default: True
    max_shape_components: None or int > 0 or 0 <= float <= 1
                          or list of those, Optional
        If list of length n_levels, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        shape components will be used for all levels.

        Per level:
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

        Default: None
    max_appearance_components: None or int > 0 or 0 <= float <= 1
                               or list of those, Opt
        If list of length n_levels, then a number of appearance components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        appearance components will be used for all levels.

        Per level:
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

        Default: None
    boundary: int >= 0, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3
    interpolator: string, Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

    Returns
    -------
    aam : :class:`menpo.fitmultiple.aam.builder.AAMBuilder`
        The AAM Builder object

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
    def __init__(self, feature_type='igo', transform=PiecewiseAffine,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=True,
                 pyramid_on_features=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3,
                 interpolator='scipy'):
        # check parameters
        self.check_n_levels(n_levels)
        self.check_downscale(downscale)
        self.check_normalization_diagonal(normalization_diagonal)
        self.check_boundary(boundary)
        max_shape_components = self.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = self.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        feature_type = self.check_feature_type(feature_type, n_levels,
                                               pyramid_on_features)

        # store parameters
        self.feature_type = feature_type
        self.transform = transform
        self.trilist = trilist
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.interpolator = interpolator

    def build(self, images, group=None, label='all', verbose=False):
        r"""
        Builds a Multilevel Active Appearance Model from a list of
        landmarked images.

        Parameters
        ----------
        images: list of :class:`menpo.image.Image`
            The set of landmarked images from which to build the AAM.
        group : string, Optional
            The key of the landmark set that should be used. If None,
            and if there is only one set of landmarks, this set will be used.

            Default: None
        label: string, Optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

            Default: 'all'
        verbose: bool, Optional
            Flag that controls information and progress printing.

            Default: False

        Returns
        -------
        aam : :class:`menpo.fitmultiple.aam.builder.AAM`
            The AAM object. Shape and appearance models are stored from lowest
            to highest level
        """
        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            self._normalization_wrt_reference_shape(
                images, group, label, self.normalization_diagonal,
                self.interpolator, verbose=verbose)

        # estimate required ram memory
        if verbose:
            self._estimate_ram_requirements(images, group, label,
                                            n_images=min([3, len(images)]))

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
        appearance_models = []

        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get images of current level
            if self.pyramid_on_features:
                # features are already computed, so just call generator
                feature_images = []
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Rescaling feature space - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    feature_images.append(g.next())
            else:
                # extract features of images returned from generator
                feature_images = []
                for c, g in enumerate(generators):
                    if verbose:
                        print_dynamic('{}Computing feature space - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                    feature_images.append(compute_features(
                        g.next(), self.feature_type[rj]))

            # format shapes to build shape model
            if j == 0:
                # extract potentially rescaled shapes
                shapes = [i.landmarks[group][label].lms
                          for i in feature_images]
            elif j != 0 and self.scaled_shape_models:
                # downscale shapes of previous level
                shapes = [Scale(1/self.downscale,
                                n_dims=shapes[0].n_dims).apply(s)
                          for s in shapes]
            # train shape model and build reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = self._build_shape_model(
                shapes, self.max_shape_components[rj])
            reference_frame = self._build_reference_frame(shape_model.mean)

            # add shape model to the list
            shape_models.append(shape_model)

            # compute transforms
            if verbose:
                print_dynamic('{}Computing transforms'.format(level_str))
            transforms = [self.transform(reference_frame.landmarks['source'].lms,
                                         i.landmarks[group][label].lms)
                          for i in feature_images]

            # warp images to reference frame
            warped_images = []
            for c, (i, t) in enumerate(zip(feature_images, transforms)):
                if verbose:
                    print_dynamic('{}Warping images - {}'.format(
                        level_str,
                        progress_bar_str(float(c + 1) / len(feature_images),
                                         show_bar=False)))
                warped_images.append(i.warp_to(reference_frame.mask, t,
                                               interpolator=self.interpolator))

            # attach reference_frame to images' source shape
            for i in warped_images:
                i.landmarks['source'] = reference_frame.landmarks['source']

            # build appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[rj] is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[rj])

            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        n_training_images = len(images)

        return self._build_aam(shape_models, appearance_models,
                               n_training_images)

    def _build_reference_frame(self, mean_shape):
        r"""
        Generates the reference frame given a mean shape.

        Parameter
        ---------
        mean_shape: Pointcloud
            The mean shape to use.

        Returns
        -------
        reference_frame : :class:`menpo.image.base.MaskedImage`
            The reference frame.
        """
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns an AAM object.

        Parameters
        ----------
        shape_models: :class:`menpo.model.pca`
            The trained multilevel shape models.
        appearance_models: :class:`menpo.model.pca`
            The trained multilevel appearance models.
        n_training_images: int
            The number of training images.

        Returns
        -------
        aam : :class:`menpo.fitmultilevel.aam.AAM`
            The trained AAM object.
        """
        return AAM(shape_models, appearance_models, n_training_images,
                   self.transform, self.feature_type, self.reference_shape,
                   self.downscale, self.scaled_shape_models,
                   self.pyramid_on_features, self.interpolator)

    def _estimate_ram_requirements(self, images, group, label, n_images=3):
        r"""
        Estimates the RAM memory requirements in order to save the AAM model
        that is about to be trained. Note that this function is only called if
        verbose parameter of the builder is enabled.

        Parameters
        ----------
        images: list of :class:`menpo.image.Image`
            The set of landmarked images which will be used to build the AAM.
        group : string
            The key of the landmark set that will be used.
        label: string
            The label of of the landmark manager that will be used.
        n_images: int, Optional
            The number of images to be used to train the temporary AAM and
            estimate the RAM requirements. Note that the images are selected
            randomly from the images list.

            Default: 3
        """
        print_dynamic('- Estimating RAM memory requirements...')
        # create images list
        n_training_images = len(images)
        which_images = sample(range(n_training_images), n_images)
        images_ram = [images[i] for i in which_images]
        # normalize images with respect to reference shape
        if self.normalization_diagonal is None:
            shapes_ram = [i.landmarks[group][label].lms for i in images_ram]
            reference_shape_ram = mean_pointcloud(shapes_ram)
            x_ram, y_ram = reference_shape_ram.range()
            x, y = self.reference_shape.range()
            scale = np.sqrt(x**2 + y**2) / np.sqrt(x_ram**2 + y_ram**2)
            Scale(scale, reference_shape_ram.n_dims).apply_inplace(
                reference_shape_ram)
            images_ram = [i.rescale_to_reference_shape(
                reference_shape_ram, group=group, label=label,
                interpolator=self.interpolator) for i in images_ram]
        # train aam
        aam_ram = self.build(images_ram, group=group, label=label,
                             verbose=False)
        # find required appearance components per level
        n_components = []
        for i in range(self.n_levels):
            if self.max_appearance_components[i] is None:
                n_components.append(n_training_images - 1)
            elif isinstance(self.max_appearance_components[i], int):
                n_components.append(
                    min([n_training_images - 1,
                         self.max_appearance_components[i]]))
            elif isinstance(self.max_appearance_components[i], float):
                n_components.append(int((n_training_images - 1) *
                                        self.max_appearance_components[i]))
        # find bytes per appearance model
        bytes_ram = [n_components[i] *
                     aam_ram.appearance_models[i].components[0, :].nbytes
                     for i in range(self.n_levels - 1, -1, -1)]
        # convert and print bytes
        print_bytes_ram = [print_bytes(i) for i in bytes_ram]
        if len(print_bytes_ram) > 1:
            print_dynamic("- Approximately {} {} of RAM required to store "
                          "model.\n".format(print_bytes(sum(bytes_ram)),
                                            print_bytes_ram))
        else:
            print_dynamic("- Approximately {} of RAM required to store "
                          "model.\n".format(print_bytes(sum(bytes_ram))))


class PatchBasedAAMBuilder(AAMBuilder):
    r"""
    Class that builds Multilevel Patch-Based Active Appearance Models.

    Parameters
    ----------
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

        Default: 'hog'
    patch_shape: tuple of ints, Optional
        The appearance model of the Patch-Based AAM will be obtained by
        sampling appearance patches with the specified shape around each
        landmark.

        Default: (16, 16)
    normalization_diagonal: int >= 20, Optional
        During building an AAM, all images are rescaled to ensure that the
        scale of their landmarks matches the scale of the mean shape.

        If int, it ensures that the mean shape is scaled so that the diagonal
        of the bounding box containing it matches the normalization_diagonal
        value.
        If None, the mean shape is not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

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
        Note that from our experience, if scaled_shape_models is False, AAMs
        tend to have slightly better performance.

        Default: True
    pyramid_on_features: boolean, Optional
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.
        Note that from our experience, if pyramid_on_features is True, AAMs
        tend to have slightly better performance.

        Default: True
    max_shape_components: None or int > 0 or 0 <= float <= 1
                          or list of those, Optional
        If list of length n_levels, then a number of shape components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        shape components will be used for all levels.

        Per level:
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

        Default: None
    max_appearance_components: None or int > 0 or 0 <= float <= 1
                               or list of those, Opt
        If list of length n_levels, then a number of appearance components is
        defined per level. The first element of the list specifies the number
        of components of the lowest pyramidal level and so on.

        If not a list or a list with length 1, then the specified number of
        appearance components will be used for all levels.

        Per level:
        If int, it specifies the exact number of components to be retained.
        If float, it specifies the percentage of variance to be retained.
        If None, all the available components are kept (100% of variance).

        Default: None
    boundary: int >= 0, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3
    interpolator: string, Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

    Returns
    -------
    aam : :class:`menpo.fitmultiple.aam.builder.PatchBasedAAMBuilder`
        The Patch-Based AAM Builder object

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
    def __init__(self, feature_type='hog', patch_shape=(16, 16),
                 normalization_diagonal=None, n_levels=3, downscale=2,
                 scaled_shape_models=True, pyramid_on_features=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3, interpolator='scipy'):
        # check parameters
        self.check_n_levels(n_levels)
        self.check_downscale(downscale)
        self.check_normalization_diagonal(normalization_diagonal)
        self.check_boundary(boundary)
        max_shape_components = self.check_max_components(
            max_shape_components, n_levels, 'max_shape_components')
        max_appearance_components = self.check_max_components(
            max_appearance_components, n_levels, 'max_appearance_components')
        feature_type = self.check_feature_type(feature_type, n_levels,
                                               pyramid_on_features)

        # store parameters
        self.feature_type = feature_type
        self.patch_shape = patch_shape
        self.normalization_diagonal = normalization_diagonal
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.interpolator = interpolator

        # patch-based AAMs can only work with TPS transform
        self.transform = ThinPlateSplines

    def _build_reference_frame(self, mean_shape):
        r"""
        Generates the reference frame given a mean shape.

        Parameter
        ---------
        mean_shape: Pointcloud
            The mean shape to use.

        Returns
        -------
        reference_frame : :class:`menpo.image.base.MaskedImage`
            The patch-based reference frame.
        """
        return build_patch_reference_frame(mean_shape, boundary=self.boundary,
                                           patch_shape=self.patch_shape)

    def _mask_image(self, image):
        r"""
        Creates the patch-based mask of the given image.

        Parameter
        ---------
        image: :class:`menpo.image.base.MaskedImage`
            The image to be masked.
        """
        image.build_mask_around_landmarks(self.patch_shape, group='source')

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns a Patch-Based AAM object.

        Parameters
        ----------
        shape_models: :class:`menpo.model.pca`
            The trained multilevel shape models.
        appearance_models: :class:`menpo.model.pca`
            The trained multilevel appearance models.
        n_training_images: int
            The number of training images.

        Returns
        -------
        aam : :class:`menpo.fitmultilevel.aam.PatchBasedAAM`
            The trained Patched-Based AAM object.
        """
        return PatchBasedAAM(shape_models, appearance_models,
                             n_training_images, self.patch_shape,
                             self.transform, self.feature_type,
                             self.reference_shape, self.downscale,
                             self.scaled_shape_models,
                             self.pyramid_on_features, self.interpolator)


class AAM(object):
    r"""
    Active Appearance Model class.

    Parameters
    -----------
    shape_models: :class:`menpo.model.PCA` list
        A list containing the shape models of the AAM.
    appearance_models: :class:`menpo.model.PCA` list
        A list containing the appearance models of the AAM.
    n_training_images: int
        The number of training images used to build the AAM.
    transform: :class:`menpo.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.
    feature_type: None or string or function/closure or list of those
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fit to
        novel images.

        If list of length n_levels, then a feature was defined per level.
        This means that the pyramid_on_features flag was disabled (False)
        and the features were extracted at each level. The first element of
        the list specifies the features of the lowest pyramidal level and so
        on.

        If not a list or a list with length 1, then:
            If pyramid_on_features is True, the specified feature was applied
            to the highest level.
            If pyramid_on_features is False, the specified feature was applied
            to all pyramid levels.

        Per level:
        If None, the appearance model was built using the original image
        representation, i.e. no features will be extracted from the original
        images.

        If string, the appearance model was built using one of Menpo's default
        built-in feature representations - those
        accessible at image.features.some_feature(). Note that this case can
        only be used with default feature parameters - for custom feature
        weights, use the functional form of this argument instead.

        If function, the user can directly provide the feature that was
        calculated on the images. This class will simply invoke this
        function, passing in as the sole argument the image to be fitted,
        and expect as a return type an Image representing the feature
        calculation ready for further fitting. See the examples for
        details.
    reference_shape: PointCloud
        The reference shape that was used to resize all training images to a
        consistent object size.
    downscale: float
        The downscale factor that was used to create the different pyramidal
        levels.
    scaled_shape_models: boolean, Optional
        If True, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.
        If False, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.
        Note that from our experience, if scaled_shape_models is False, AAMs
        tend to have slightly better performance.
    pyramid_on_features: boolean, Optional
        If True, the feature space was computed once at the highest scale and
        the Gaussian pyramid was applied on the feature images.
        If False, the Gaussian pyramid was applied on the original images
        (intensities) and then features were extracted at each level.
        Note that from our experience, if pyramid_on_features is True, AAMs
        tend to have slightly better performance.
    interpolator: string
        The interpolator that was used to build the AAM.
    """
    def __init__(self, shape_models, appearance_models, n_training_images,
                 transform, feature_type, reference_shape, downscale,
                 scaled_shape_models, pyramid_on_features, interpolator):
        self.n_training_images = n_training_images
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.feature_type = feature_type
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features
        self.interpolator = interpolator

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the AAM.

        :type: int
        """
        return len(self.appearance_models)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
        returned.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or float list
            Weights of the shape model that will be used to create
            a novel shape instance. If None, the mean shape
            (shape_weights = [0, 0, ..., 0]) is used.

            Default: None
        appearance_weights: (n_weights,) ndarray or float list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If None, the mean appearance
            (appearance_weights = [0, 0, ..., 0]) is used.

            Default: None
        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`menpo.image.base.Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)
        n_appearance_weights = len(appearance_weights)
        appearance_weights *= am.eigenvalues[:n_appearance_weights] ** 0.5
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

    def random_instance(self, level=-1):
        r"""
        Generates a novel random instance of the AAM.

        Parameters
        -----------
        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`menpo.image.base.Image`
            The novel AAM instance.
        """
        sm = self.shape_models[level]
        am = self.appearance_models[level]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)
        appearance_weights = (np.random.randn(am.n_active_components) *
                              am.eigenvalues[:am.n_active_components]**0.5)
        appearance_instance = am.instance(appearance_weights)

        return self._instance(level, shape_instance, appearance_instance)

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance, landmarks)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.warp_to(
            reference_frame.mask, transform, self.interpolator)

    def _build_reference_frame(self, reference_shape, landmarks):
        if type(landmarks) == TriMesh:
            trilist = landmarks.trilist
        else:
            trilist = None
        return build_reference_frame(
            reference_shape, trilist=trilist)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        : str
        """
        return 'Active Appearance Model'

    def __str__(self):
        out = "{}\n - {} training images.\n".format(self._str_title,
                                                    self.n_training_images)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        for j in range(self.n_levels):
            n_channels.append(
                self.appearance_models[j].template_instance.n_channels)
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
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
                ch_str = "channel"
            else:
                ch_str = "channels"
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
        out = "{} - Warp using {} transform with '{}' interpolation.\n".format(
            out, self.transform.__name__, self.interpolator)
        if self.n_levels > 1:
            if self.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n".format(out, self.n_levels,
                                                          self.downscale)

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n".format(out, self.n_levels,
                                                 self.downscale)
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str)
                if self.scaled_shape_models is False:
                    out = "{}   - Reference frames of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out, self.appearance_models[0].n_features,
                          self.appearance_models[0].template_instance.n_true_pixels,
                          n_channels[0],
                          self.appearance_models[0].template_instance._str_shape,
                          n_channels[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if self.pyramid_on_features is False:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                if (self.scaled_shape_models or
                        self.pyramid_on_features is False):
                    out = "{}     - Reference frame of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out, self.appearance_models[i].n_features,
                          self.appearance_models[i].template_instance.n_true_pixels,
                          n_channels[i],
                          self.appearance_models[i].template_instance._str_shape,
                          n_channels[i])
                out = "{0}     - {1} shape components ({2:.2f}% of " \
                      "variance)\n     - {3} appearance components " \
                      "({4:.2f}% of variance)\n".format(
                      out, self.shape_models[i].n_components,
                      self.shape_models[i].variance_ratio * 100,
                      self.appearance_models[i].n_components,
                      self.appearance_models[i].variance_ratio * 100)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - Reference frame of length {4} ({5} x {6}C, " \
                  "{7} x {8}C)\n   - {9} shape components ({10:.2f}% of " \
                  "variance)\n   - {11} appearance components ({12:.2f}% of " \
                  "variance)\n".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  self.appearance_models[0].n_features,
                  self.appearance_models[0].template_instance.n_true_pixels,
                  n_channels[0],
                  self.appearance_models[0].template_instance._str_shape,
                  n_channels[0], self.shape_models[0].n_components,
                  self.shape_models[0].variance_ratio * 100,
                  self.appearance_models[0].n_components,
                  self.appearance_models[0].variance_ratio * 100)
        return out


class PatchBasedAAM(AAM):
    r"""
    Patch Based Active Appearance Model class.

    Parameters
    -----------
    shape_models: :class:`menpo.model.PCA` list
        A list containing the shape models of the AAM.
    appearance_models: :class:`menpo.model.PCA` list
        A list containing the appearance models of the AAM.
    n_training_images: int
        The number of training images used to build the AAM.
    patch_shape: tuple of ints
        The shape of the patches used to build the Patch Based AAM.
    transform: :class:`menpo.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.
    feature_type: None or string or function/closure or list of those
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fit to
        novel images.

        If list of length n_levels, then a feature was defined per level.
        This means that the pyramid_on_features flag was disabled (False)
        and the features were extracted at each level. The first element of
        the list specifies the features of the lowest pyramidal level and so
        on.

        If not a list or a list with length 1, then:
            If pyramid_on_features is True, the specified feature was applied
            to the highest level.
            If pyramid_on_features is False, the specified feature was applied
            to all pyramid levels.

        Per level:
        If None, the appearance model was built using the original image
        representation, i.e. no features will be extracted from the original
        images.

        If string, the appearance model was built using one of Menpo's default
        built-in feature representations - those
        accessible at image.features.some_feature(). Note that this case can
        only be used with default feature parameters - for custom feature
        weights, use the functional form of this argument instead.

        If function, the user can directly provide the feature that was
        calculated on the images. This class will simply invoke this
        function, passing in as the sole argument the image to be fitted,
        and expect as a return type an Image representing the feature
        calculation ready for further fitting. See the examples for
        details.
    reference_shape: PointCloud
        The reference shape that was used to resize all training images to a
        consistent object size.
    downscale: float
        The downscale factor that was used to create the different pyramidal
        levels.
    scaled_shape_models: boolean, Optional
        If True, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.
        If False, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.
        Note that from our experience, if scaled_shape_models is False, AAMs
        tend to have slightly better performance.
    pyramid_on_features: boolean, Optional
        If True, the feature space was computed once at the highest scale and
        the Gaussian pyramid was applied on the feature images.
        If False, the Gaussian pyramid was applied on the original images
        (intensities) and then features were extracted at each level.
        Note that from our experience, if pyramid_on_features is True, AAMs
        tend to have slightly better performance.
    interpolator: string
        The interpolator that was used to build the AAM.
    """
    def __init__(self, shape_models, appearance_models, n_training_images,
                 patch_shape, transform, feature_type, reference_shape,
                 downscale, scaled_shape_models, pyramid_on_features,
                 interpolator):
        super(PatchBasedAAM, self).__init__(
            shape_models, appearance_models, n_training_images, transform,
            feature_type, reference_shape, downscale, scaled_shape_models,
            pyramid_on_features, interpolator)
        self.patch_shape = patch_shape

    def _build_reference_frame(self, reference_shape, landmarks):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        : str
        """
        return 'Patch-Based Active Appearance Model'

    def __str__(self):
        out = super(PatchBasedAAM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        out_splitted.insert(5, "   - Patch size is {}W x {}H.".format(
            self.patch_shape[1], self.patch_shape[0]))
        return '\n'.join(out_splitted)


def build_reference_frame(landmarks, boundary=3, group='source',
                          trilist=None):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks: PointCloud
        The landmarks that will be used to build the reference frame.
    boundary: int, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3
    group: str, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

        Default: 'source'
    trilist: (t, 3) ndarray, Optional
        Triangle list that will be used to build the reference frame. If None,
        defaults to performing Delaunay triangulation on the points.

        Default: None

    Returns
    -------
    reference_frame : :class:`menpo.image.base.Image`
        The reference frame.
    """
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)
    if trilist is not None:
        reference_frame.landmarks[group] = TriMesh(
            reference_frame.landmarks['source'].lms.points, trilist=trilist)

    # TODO: revise kwarg trilist in method constrain_mask_to_landmarks,
    # perhaps the trilist should be directly obtained from the group landmarks
    reference_frame.constrain_mask_to_landmarks(group=group, trilist=trilist)

    return reference_frame


def build_patch_reference_frame(landmarks, boundary=3, group='source',
                                patch_shape=(16, 16)):
    r"""
    Builds a reference frame from a particular set of landmarks.

    Parameters
    ----------
    landmarks: PointCloud
        The landmarks that will be used to build the reference frame.
    boundary: int, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3
    group: str, optional
        Group that will be assigned to the provided set of landmarks on the
        reference frame.

        Default: 'source'
    patch_shape: tuple of ints, Optional
        Tuple specifying the shape of the patches.

        Default: (16, 16)

    Returns
    -------
    patch_based_reference_frame : :class:`menpo.image.base.Image`
        The patch based reference frame.
    """
    boundary = np.max(patch_shape) + boundary
    reference_frame = _build_reference_frame(landmarks, boundary=boundary,
                                             group=group)

    # mask reference frame
    reference_frame.build_mask_around_landmarks(patch_shape, group=group)

    return reference_frame


def _build_reference_frame(landmarks, boundary=3, group='source'):
    # translate landmarks to the origin
    minimum = landmarks.bounds(boundary=boundary)[0]
    landmarks = Translation(-minimum).apply(landmarks)

    resolution = landmarks.range(boundary=boundary)
    reference_frame = MaskedImage.blank(resolution)
    reference_frame.landmarks[group] = landmarks

    return reference_frame
