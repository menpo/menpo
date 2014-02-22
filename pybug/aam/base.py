from __future__ import division
import numpy as np
from pybug.shape import TriMesh
from pybug.transform .affine import Scale, UniformScale, Translation
from pybug.groupalign import GeneralizedProcrustesAnalysis
from pybug.transform.piecewiseaffine import PiecewiseAffineTransform
from pybug.transform.tps import TPS
from pybug.model import PCAModel
from pybug.aam.functions import \
    mean_pointcloud, build_reference_frame, build_patch_reference_frame, \
    compute_features


class AAM(object):
    r"""
    Active Appearance Model (AAM) class.

    Parameters
    -----------
    shape_models: :class:`pybug.model.PCA` list
        A list containing the shape models of the AAM.

    appearance_models: :class:`pybug.model.PCA` list
        A list containing the appearance models of the AAM.

    transform_cls: :class:`pybug.transform.PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    feature_type: str or closure
        The feature_type used to build the AAM.

        If None, the appearance model was built using the original image
        representation, i.e. no features were extracted from the original
        image representation.

        If string or closure, the appearance model was built from
        a feature representation of the original images:
            If string, the was was built using the following feature
            computation:

               feature_image = eval('img.feature_type.' + feature_type + '()')

            For this to work properly the feature_type needs to be one of
            Pybug's standard image feature methods. Note that, in this case,
            the feature computation was carried out using its default options.

            Non-default feature options and new experimental feature could
            have been used through a closure definition. In this case,
            the closure must have defined a function that receives as an
            input an image and returns a particular feature representation
            of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `pybug.image.MaskedNDImage` for details more details on Pybug's
            standard image features and feature options.

    interpolator:'scipy' or 'cinterp' or func
        The interpolator used by the previous warps.

    downscale: float
        The downscale factor used to create the different levels of the AAM.

    patch_size: integer tuple or None
        Tuple specifying the size of the patches used to build the AAM . If
        None, the AAM was not build using a Patch-Based representation.
    """

    def __init__(self, shape_models, appearance_models, transform_cls,
                 feature_type, reference_shape, downscale, patch_size,
                 interpolator):
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform_cls = transform_cls
        self.feature_type = feature_type
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.patch_size = patch_size
        self.interpolator = interpolator

        if len(appearance_models) > 1:
            difference = UniformScale.align(
                appearance_models[0].mean.landmarks['source'].lms,
                appearance_models[1].mean.landmarks['source'].lms).as_vector()
            if difference == 1:
                self.scaled_reference_frames = False
            else:
                self.scaled_reference_frames = True
        else:
            self.downscale = None
            self.scaled_reference_frames = None

    @property
    def n_levels(self):
        return len(self.appearance_models)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Generates a novel AAM instance.

        Parameters
        -----------
        shape_weights: (n_weights,) ndarray or float list
            Weights of the shape model that will be used to create
            a novel shape instance. If None, the mean shape
            (shape_weights = [0, 0, ..., 0]) will be used.

            Default: None
        appearance_weights: (n_weights,) ndarray or float list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If None, the mean appearance
            (appearance_weights = [0, 0, ..., 0]) will be used.

            Default: None
        level: int, optional
            The pyramidal level to be used.

            Default: -1

        Returns
        -------
        image: :class:`pybug.image.masked.MaskedImage`
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
        image: :class:`pybug.image.masked.MaskedImage`
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

        # build reference frame
        if self.patch_size:
            reference_frame = build_patch_reference_frame(
                shape_instance, patch_size=self.patch_size)
        else:
            if type(landmarks) == TriMesh:
                trilist = template.landmarks['source'].lms.trilist
            else:
                trilist = None
            reference_frame = build_reference_frame(
                shape_instance, trilist=trilist)

        transform = self.transform_cls(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.warp_to(reference_frame.mask, transform,
                                           self.interpolator)


def aam_builder(images, group=None, label='all', interpolator='scipy',
                diagonal_range=None, boundary=3,
                transform_cls=PiecewiseAffineTransform,
                trilist=None, patch_size=None, n_levels=3, downscale=2,
                scaled_reference_frames=False, feature_type=None,
                max_shape_components=None, max_appearance_components=None):

    r"""
    Builds an AAM object from a set of landmark images.

    Parameters
    ----------
    images: list of :class:`pybug.image.IntensityImage`
        The set of landmarked images from which to build the AAM

    group : string, Optional
        The key of the landmark set that should be used. If None,
        and if there is only one set of landmarks, this set will be used.

        Default: None

    label: string, Optional
        The label of of the landmark manager that you wish to use. If no
        label is passed, the convex hull of all landmarks is used.

        Default: 'all'

    interpolator:'scipy' or 'cinterp' or func, Optional
        The interpolator that should be used to perform the warps.

        Default: 'scipy'

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

    boundary: int, Optional
        The number of pixels to be left as a "save" margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).

        Default: 3

    transform_cls: :class:`pybug.transform.PureAlignmentTransform`, Optional
        The :class:`pybug.transform.PureAlignmentTransform` that will be
        used to warp the images.

        Default: PieceWiseAffine

    trilist: (t, 3) ndarray, Optional
        Triangle list that will be used to build the reference frame. If None,
        defaults to performing Delaunay triangulation on the points.

        Default: None

        .. note::

            This kwarg will be completely ignored if the kwarg transform_cls
            is not set :class:`pybug.transform.PiecewiseAffineTransform` or
            if the kwarg patch_size is not set to None.

    patch_size: tuple of ints or None, Optional
        If tuple, the appearance model of the AAM will be obtained from by
        sampling the appearance patches around the landmarks. If None, the
        standard representation for the AAMs' appearance model will be used
        instead.

        Default: None

        .. note::

            If tuple, the kwarg transform_cls will be automatically set to
            :class:`pybug.transform.TPS`.

    n_levels: int, Optional
        The number of multi-resolution pyramidal levels to be used.

        Default: 3

    downscale: float > 1, Optional
        The downscale factor that will be used to create the different AAM
        pyramidal levels.

        Default: 2

    scaled_reference_frames: boolean, Optional
        If False, the resolution of all reference frame used to build the
        appearance model will be fixed (the original images will be
        both smooth and scaled using a Gaussian pyramid). consequently, all
        appearance models will have the same dimensionality.
        If True, the reference frames used to create the appearance model
        will be themselves scaled (the original images will only be smooth).
        Consequently the dimensionality of all appearance models will be
        different.

        Default: False

    feature_type: string or closure, Optional
        If None, the appearance model will be build using the original image
        representation, i.e. no features will be extracted from the original
        images.
        If string or closure, the appearance model will be build from a
        feature representation of the original images:
            If string, the `ammbuilder` will try to compute image features by
            executing:

               feature_image = eval('img.feature_type.' + feature_type + '()')

            For this to work properly the feature_type needs to be one of
            Pybug's standard image feature methods. Note that, in this case,
            the feature computation will be carried out using its default
            options.

            Non-default feature options and new experimental feature can be
            used by defining a closure. In this case, the closure must define a
            function that receives as an input an image and returns a
            particular feature representation of that image. For example:

                def igo_double_from_std_normalized_intensities(image)
                    image = deepcopy(image)
                    image.normalize_std_inplace()
                    return image.feature_type.igo(double_angles=True)

            See `pybug.image.MaskedNDImage` for details more details on Pybug's
            standard image features and feature options.

        Default: None

    max_shape_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original shape model to be retained.

        Default: None

    max_appearance_components: 0 < int < n_components, Optional
        If int, it specifies the specific number of components of the
        original appearance model to be retained.

        Default: None

    Returns
    -------
    aam : :class:`pybug.aam.AAM`
        The AAM object
    """

    if patch_size is not None:
        transform_cls = TPS

    print '- Rescaling images'
    shapes = [i.landmarks[group][label].lms for i in images]
    reference_shape = mean_pointcloud(shapes)
    if diagonal_range:
        x, y = reference_shape.range()
        scale = diagonal_range / np.sqrt(x**2 + y**2)
        Scale(scale, reference_shape.n_dims).apply_inplace(reference_shape)
    images = [i.rescale_to_reference_landmarks(reference_shape,
                                               group=group, label=label,
                                               interpolator=interpolator)
              for i in images]

    if scaled_reference_frames:
        print '- Setting gaussian smoothing generators'
        generator = [i.smoothing_pyramid(n_levels=n_levels,
                                         downscale=downscale)
                     for i in images]
    else:
        print '- Setting gaussian pyramid generators'
        generator = [i.gaussian_pyramid(n_levels=n_levels,
                                        downscale=downscale)
                     for i in images]

    print '- Building model pyramids'
    shape_models = []
    appearance_models = []
    # for each level
    for j in np.arange(n_levels):
        print ' - Level {}'.format(j)

        print '  - Computing feature_type'
        images = [compute_features(g.next(), feature_type) for g in generator]
        # extract potentially rescaled shapes
        shapes = [i.landmarks[group][label].lms for i in images]

        if scaled_reference_frames or j == 0:
            print '  - Building shape model'
            if j != 0:
                shapes = [Scale(1/downscale, n_dims=shapes[0].n_dims).apply(s)
                          for s in shapes]
            # centralize shapes
            centered_shapes = [Translation(-s.centre).apply(s) for s in shapes]
            # align centralized shape using Procrustes Analysis
            gpa = GeneralizedProcrustesAnalysis(centered_shapes)
            aligned_shapes = [s.aligned_source for s in gpa.transforms]

            # build shape model
            shape_model = PCAModel(aligned_shapes)
            if max_shape_components is not None:
                # trim shape model if required
                shape_model.trim_components(max_shape_components)

            print '  - Building reference frame'
            mean_shape = mean_pointcloud(aligned_shapes)
            if patch_size is not None:
                # build patch based reference frame
                reference_frame = build_patch_reference_frame(
                    mean_shape, boundary=boundary, patch_size=patch_size)
            else:
                # build reference frame
                reference_frame = build_reference_frame(
                    mean_shape, boundary=boundary, trilist=trilist)

        # add shape model to the list
        shape_models.append(shape_model)

        print '  - Computing transforms'
        transforms = [transform_cls(reference_frame.landmarks['source'].lms,
                                    i.landmarks[group][label].lms)
                      for i in images]

        print '  - Warping images'
        images = [i.warp_to(reference_frame.mask, t,
                            interpolator=interpolator)
                  for i, t in zip(images, transforms)]

        for i in images:
            i.landmarks['source'] = reference_frame.landmarks['source']
        if patch_size:
            for i in images:
                i.build_mask_around_landmarks(patch_size, group='source')
        else:
            for i in images:
                i.constrain_mask_to_landmarks(group='source', trilist=trilist)

        print '  - Building appearance model'
        appearance_model = PCAModel(images)
        # trim appearance model if required
        if max_appearance_components is not None:
            appearance_model.trim_components(max_appearance_components)

        # add appearance model to the list
        appearance_models.append(appearance_model)

    # reverse the list of shape and appearance models so that they are
    # ordered from lower to higher resolution
    shape_models.reverse()
    appearance_models.reverse()

    return AAM(shape_models, appearance_models, transform_cls, feature_type,
               reference_shape, downscale, patch_size, interpolator)
