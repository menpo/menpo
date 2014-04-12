from __future__ import division
import numpy as np

from menpo.shape import TriMesh
from menpo.image import MaskedImage
from menpo.transform import Scale, Translation
from menpo.transform.piecewiseaffine import PiecewiseAffine
from menpo.transform.thinplatesplines import ThinPlateSplines
from menpo.model import PCAModel
from menpo.fitmultilevel.builder import DeformableModelBuilder
from menpo.fitmultilevel.featurefunctions import compute_features
from menpo.fitmultilevel.featurefunctions import sparse_hog


class AAMBuilder(DeformableModelBuilder):
    r"""
    Class that builds Multilevel Active Appearance Models.

    Parameters
    ----------
    feature_type: string or function/closure, Optional
        If None, the appearance model will be build using the original image
        representation, i.e. no features will be extracted from the original
        images.
        If string or closure, the appearance model will be built from a
        feature representation of the original images:
            If string, image features will be computed by executing:

               feature_image = eval('img.feature_type.' + feature_type + '()')

            For this to work properly `feature_type` needs to be one of
            menpo's standard image feature methods. Note that, in this case,
            the feature computation will be carried out using the default
            options.

            Non-default feature options and new experimental features can be
            defined using functions/closures. In this case, the function must
            receive an image as input and return a particular feature
            representation of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `menpo.image.feature.py` for details more details on
            menpo's standard image features and feature options.

        Default: None

    transform: :class:`menpo.transform.PureAlignmentTransform`, Optional
        The :class:`menpo.transform.PureAlignmentTransform` that will be
        used to warp the images.

        Default: :class:`menpo.transform.PiecewiseAffine`

    trilist: (t, 3) ndarray, Optional
        Triangle list that will be used to build the reference frame. If None,
        defaults to performing Delaunay triangulation on the points.

        Default: None

    diagonal_range: int, Optional
        All images will be rescaled to ensure that the scale of their
        landmarks matches the scale of the mean shape.

        If int, ensures that the mean shape is scaled so that
        the diagonal of the bounding box containing it matches the
        diagonal_range value.
        If None, the mean landmarks are not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

        Default: None

    n_levels: int, Optional
        The number of multi-resolution pyramidal levels to be used.

        Default: 3

    downscale: float > 1, Optional
        The downscale factor that will be used to create the different
        pyramidal levels.

        Default: 2

    scaled_levels: boolean, Optional
        If True, the original images will be both smoothed and scaled using
        a Gaussian pyramid to create the different pyramidal levels.
        If False, they will only be smoothed.

        Default: True

    max_shape_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original shape model to be retained.

        Default: None

    max_appearance_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original appearance model to be retained.

        Default: None

    boundary: int, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3

    interpolator:'scipy', Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

    Returns
    -------
    aam : :class:`menpo.fitmultiple.aam.builder.AAMBuilder`
        The AAM Builder object
    """
    def __init__(self, feature_type=sparse_hog,
                 transform=PiecewiseAffine, trilist=None,
                 diagonal_range=None, n_levels=3, downscale=1.1,
                 scaled_levels=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3,
                 interpolator='scipy'):
        self.feature_type = feature_type
        self.transform = transform
        self.trilist = trilist
        self.diagonal_range = diagonal_range
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_levels = scaled_levels
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.interpolator = interpolator

    def build(self, images, group=None, label='all'):
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

        Returns
        -------
        aam : :class:`menpo.fitmultiple.aam.builder.AAM`
            The AAM object
        """
        print '- Preprocessing'
        self.reference_shape, generator = self._preprocessing(
            images, group, label, self.diagonal_range, self.interpolator,
            self.scaled_levels, self.n_levels, self.downscale)

        print '- Building model pyramids'
        shape_models = []
        appearance_models = []
        # for each level
        for j in np.arange(self.n_levels):
            print ' - Level {}'.format(j)

            print '  - Computing feature space'
            images = [compute_features(g.next(), self.feature_type)
                      for g in generator]
            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label].lms for i in images]

            if j == 0 or self.scaled_levels:
                print '  - Building shape model'
                if j != 0:
                    shapes = [Scale(1/self.downscale,
                                    n_dims=shapes[0].n_dims).apply(s)
                              for s in shapes]
                shape_model = self._build_shape_model(
                    shapes, self.max_shape_components)

                print '  - Building reference frame'
                reference_frame = self._build_reference_frame(
                    shape_model.mean)

            # add shape model to the list
            shape_models.append(shape_model)

            print '  - Computing transforms'
            transforms = [self.transform(reference_frame.landmarks['source'].lms,
                                         i.landmarks[group][label].lms)
                          for i in images]

            print '  - Warping images'
            images = [i.warp_to(reference_frame.mask, t,
                                interpolator=self.interpolator)
                      for i, t in zip(images, transforms)]

            for i in images:
                i.landmarks['source'] = reference_frame.landmarks['source']
                self._mask_image(i)

            print '  - Building appearance model'
            appearance_model = PCAModel(images)
            # trim appearance model if required
            if self.max_appearance_components is not None:
                appearance_model.trim_components(
                    self.max_appearance_components)

            # add appearance model to the list
            appearance_models.append(appearance_model)

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()

        return self._build_aam(shape_models, appearance_models)

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary,
                                     trilist=self.trilist)

    def _mask_image(self, image):
        image.constrain_mask_to_landmarks(group='source',
                                          trilist=self.trilist)

    def _build_aam(self, shape_models, appearance_models):
        return AAM(shape_models, appearance_models, self.transform,
                   self.feature_type, self.reference_shape, self.downscale,
                   self.scaled_levels, self.interpolator)


#TODO: Test me!!!
class PatchBasedAAMBuilder(AAMBuilder):
    r"""
    Class that builds Patch-Based Multilevel Active Appearance Models.

    Parameters
    ----------
    feature_type: string or function/closure, Optional
        If None, the appearance model will be build using the original image
        representation, i.e. no features will be extracted from the original
        images.
        If string or closure, the appearance model will be built from a
        feature representation of the original images:
            If string, image features will be computed by executing:

               feature_image = eval('img.feature_type.' + feature_type + '()')

            For this to work properly `feature_type` needs to be one of
            menpo's standard image feature methods. Note that, in this case,
            the feature computation will be carried out using the default
            options.

            Non-default feature options and new experimental features can be
            defined using functions/closures. In this case, the function must
            receive an image as input and return a particular feature
            representation of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `menpo.image.feature.py` for details more details on
            menpo's standard image features and feature options.

        Default: None

    transform: :class:`menpo.transform.PureAlignmentTransform`, Optional
        The :class:`menpo.transform.PureAlignmentTransform` that will be
        used to warp the images.

        Default: :class:`menpo.transform.PiecewiseAffine`

    patch_shape: tuple of ints, Optional
        The appearance model of the Patch-Based AAM will be obtained by
        sampling appearance patches with the specified shape around each
        landmark.

        Default: (16, 16)

    diagonal_range: int, Optional
        All images will be rescaled to ensure that the scale of their
        landmarks matches the scale of the mean shape.

        If int, ensures that the mean shape is scaled so that
        the diagonal of the bounding box containing it matches the
        diagonal_range value.
        If None, the mean landmarks are not rescaled.

        Note that, because the reference frame is computed from the mean
        landmarks, this kwarg also specifies the diagonal length of the
        reference frame (provided that features computation does not change
        the image size).

        Default: None

    n_levels: int, Optional
        The number of multi-resolution pyramidal levels to be used.

        Default: 3

    downscale: float > 1, Optional
        The downscale factor that will be used to create the different
        pyramidal levels.

        Default: 2

    scaled_levels: boolean, Optional
        If True, the original images will be both smoothed and scaled using
        a Gaussian pyramid to create the different pyramidal levels.
        If False, they will only be smoothed.

        Default: True

    max_shape_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original shape model to be retained.

        Default: None

    max_appearance_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original appearance model to be retained.

        Default: None

    boundary: int, Optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3

    interpolator:'scipy', Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

    Returns
    -------
    aam : :class:`menpo.fitmultiple.aam.builder.PatchBasedAAMBuilder`
        The Patch Based AAM Builder object
    """
    def __init__(self, feature_type='hog', transform=ThinPlateSplines,
                 patch_shape=(16, 16), diagonal_range=None, n_levels=3,
                 downscale=2, scaled_levels=True, max_shape_components=None,
                 max_appearance_components=None, boundary=3,
                 interpolator='scipy'):
        self.feature_type = feature_type
        self.transform = transform
        self.patch_shape = patch_shape
        self.diagonal_range = diagonal_range
        self.n_levels = n_levels
        self.downscale = downscale
        self.scaled_levels = scaled_levels
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.boundary = boundary
        self.interpolator = interpolator

    def _build_reference_frame(self, mean_shape):
        return build_patch_reference_frame(mean_shape, boundary=self.boundary,
                                           patch_shape=self.patch_shape)

    def _mask_image(self, image):
        image.build_mask_around_landmarks(self.patch_shape, group='source')

    def _build_aam(self, shape_models, appearance_models):
        return PatchBasedAAM(shape_models, appearance_models,
                             self.patch_shape, self.transform,
                             self.feature_type, self.patch_shape,
                             self.reference_shape, self.downscale,
                             self.interpolator)


class AAM(object):
    r"""
    Active Appearance Model class.

    Parameters
    -----------
    shape_models: :class:`menpo.model.PCA` list
        A list containing the shape models of the AAM.

    appearance_models: :class:`menpo.model.PCA` list
        A list containing the appearance models of the AAM.

    transform: :class:`menpo.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    feature_type: str or function
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fitter to
        novel images.

        If None, the appearance model was built immediately from the image
        representation, i.e. intensity.

        If string, the appearance model was built using one of Menpo's default
        built-in feature representations - those
        accessible at image.features.some_feature(). Note that this case can
        only be used with default feature weights - for custom feature
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
        The constant downscale factor used to create the different levels of
        the AAM. For example, a factor of 2 would imply that the second level
        of the AAM pyramid is half the width and half the height of the first.
        The third would be 1/2 * 1/2 = 1/4 the width and 1/4 the height of
        the original.

    scaled_levels: boolean
        Boolean value specifying whether the AAM levels are scaled or not.

    interpolator: string
        The interpolator that was used to build the AAM.

        Default: 'scipy'
    """
    def __init__(self, shape_models, appearance_models, transform,
                 feature_type, reference_shape, downscale, scaled_levels,
                 interpolator):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.feature_type = feature_type
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_levels = scaled_levels
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


#TODO: Test me!!!
class PatchBasedAAM(AAM):
    r"""
    Patch Based Active Appearance Model class.

    Parameters
    -----------
    shape_models: :class:`menpo.model.PCA` list
        A list containing the shape models of the AAM.

    appearance_models: :class:`menpo.model.PCA` list
        A list containing the appearance models of the AAM.

    patch_shape: tuple of ints
        The shape of the patches used to build the Patch Based AAM.

    transform: :class:`menpo.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    feature_type: str or function
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fitter to
        novel images.

        If None, the appearance model was built immediately from the image
        representation, i.e. intensity.

        If string, the appearance model was built using one of Menpo's default
        built-in feature representations - those
        accessible at image.features.some_feature(). Note that this case can
        only be used with default feature weights - for custom feature
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
        The constant downscale factor used to create the different levels of
        the AAM. For example, a factor of 2 would imply that the second level
        of the AAM pyramid is half the width and half the height of the first.
        The third would be 1/2 * 1/2 = 1/4 the width and 1/4 the height of
        the original.

    scaled_levels: boolean
        Boolean value specifying whether the AAM levels are scaled or not.

    interpolator: string
        The interpolator that was used to build the AAM.

        Default: 'scipy'
    """
    def __init__(self, shape_models, appearance_models, patch_shape,
                 transform, feature_type, reference_shape, downscale,
                 scaled_levels, interpolator):
        super(PatchBasedAAM, self).__init__(
            shape_models, appearance_models, transform, feature_type,
            reference_shape, downscale, scaled_levels, interpolator)
        self.patch_shape = patch_shape

    def _build_reference_frame(self, reference_shape, landmarks):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)


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
