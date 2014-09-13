from __future__ import division

import numpy as np
from hdf5able import HDF5able, SerializableCallable

from menpo.shape import TriMesh
from .builder import build_patch_reference_frame, build_reference_frame


class AAM(HDF5able):
    r"""
    Active Appearance Model class.

    Parameters
    -----------
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.

    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.

    n_training_images : `int`
        The number of training images used to build the AAM.

    transform : :map:`PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    features : ``None`` or `string` or `function` or list of those
        The image feature that was be used to build the ``appearance_models``.
        Will subsequently be used by fitter objects using this class to fit to
        novel images.

        If list of length ``n_levels``, then a feature was defined per level.
        This means that the ``pyramid_on_features`` flag was ``False``
        and the features were extracted at each level. The first element of
        the list specifies the features of the lowest pyramidal level and so
        on.

        If not a list or a list with length ``1``, then:
            If ``pyramid_on_features`` is ``True``, the specified feature was
            applied to the highest level.

            If ``pyramid_on_features`` is ``False``, the specified feature was
            applied to all pyramid levels.

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

        Note that from our experience, if scaled_shape_models is ``False``, AAMs
        tend to have slightly better performance.

    pyramid_on_features : `boolean`, optional
        If ``True``, the feature space was computed once at the highest scale
        and the Gaussian pyramid was applied on the feature images.

        If ``False``, the Gaussian pyramid was applied on the original images
        (intensities) and then features were extracted at each level.

        Note that from our experience, if ``pyramid_on_features`` is ``True``,
        AAMs tend to have slightly better performance.

    """
    def __init__(self, shape_models, appearance_models, n_training_images,
                 transform, features, reference_shape, downscale,
                 scaled_shape_models, pyramid_on_features):
        self.n_training_images = n_training_images
        self.shape_models = shape_models
        self.appearance_models = appearance_models
        self.transform = transform
        self.features = features
        self.reference_shape = reference_shape
        self.downscale = downscale
        self.scaled_shape_models = scaled_shape_models
        self.pyramid_on_features = pyramid_on_features

    def h5_dict_to_serializable_dict(self):
        import menpo.transform
        d = self.__dict__.copy()
        transform = d.pop('transform')
        d['transform'] = SerializableCallable(transform, [menpo.transform])

        features = d.pop('features')
        d['features'] = [SerializableCallable(f, [menpo.feature])
                         for f in features]
        return d

    @property
    def n_levels(self):
        """
        The number of multi-resolution pyramidal levels of the AAM.

        :type: `int`
        """
        return len(self.appearance_models)

    def instance(self, shape_weights=None, appearance_weights=None, level=-1):
        r"""
        Generates a novel AAM instance given a set of shape and appearance
        weights. If no weights are provided, the mean AAM instance is
        returned.

        Parameters
        -----------
        shape_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the shape model that will be used to create
            a novel shape instance. If ``None``, the mean shape
            ``(shape_weights = [0, 0, ..., 0])`` is used.

        appearance_weights : ``(n_weights,)`` `ndarray` or `float` list
            Weights of the appearance model that will be used to create
            a novel appearance instance. If ``None``, the mean appearance
            ``(appearance_weights = [0, 0, ..., 0])`` is used.

        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
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
        level : `int`, optional
            The pyramidal level to be used.

        Returns
        -------
        image : :map:`Image`
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

        return appearance_instance.warp_to_mask(reference_frame.mask,
                                                transform)

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

        :type: `string`
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
            if isinstance(self.features[0], str):
                feat_str = "- Feature is {} with ".format(
                    self.features[0])
            elif self.features[0] is None:
                feat_str = "- No features extracted. "
            else:
                feat_str = "- Feature is {} with ".format(
                    self.features[0].__name__)
            if n_channels[0] == 1:
                ch_str = ["channel"]
            else:
                ch_str = ["channels"]
        else:
            feat_str = []
            ch_str = []
            for j in range(self.n_levels):
                if isinstance(self.features[j], str):
                    feat_str.append("- Feature is {} with ".format(
                        self.features[j]))
                elif self.features[j] is None:
                    feat_str.append("- No features extracted. ")
                else:
                    feat_str.append("- Feature is {} with ".format(
                        self.features[j].__name__))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        out = "{} - Warp.\n".format(out, self.transform.__name__)
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
                                                    n_channels[0], ch_str[0])
                if not self.scaled_shape_models:
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
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                if (self.scaled_shape_models or
                        (not self.pyramid_on_features)):
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
    shape_models : :map:`PCAModel` list
        A list containing the shape models of the AAM.

    appearance_models : :map:`PCAModel` list
        A list containing the appearance models of the AAM.

    n_training_images : `int`
        The number of training images used to build the AAM.

    patch_shape : tuple of `int`
        The shape of the patches used to build the Patch Based AAM.

    transform : :map:`PureAlignmentTransform`
        The transform used to warp the images from which the AAM was
        constructed.

    features : `function` or list of those
        The image feature that was be used to build the appearance_models. Will
        subsequently be used by fitter objects using this class to fit to
        novel images.

        If list of length ``n_levels``, then a feature was defined per level.
        This means that the ``pyramid_on_features`` flag was ``False``
        and the features were extracted at each level. The first element of
        the list specifies the features of the lowest pyramidal level and so
        on.

        If not a list or a list with length ``1``, then:
            If ``pyramid_on_features`` is ``True``, the specified feature was
            applied to the highest level.

            If ``pyramid_on_features`` is ``False``, the specified feature was
            applied to all pyramid levels.


    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that was used to create the different pyramidal
        levels.

    scaled_shape_models : `boolean`, optional
        If ``True``, the reference frames are the mean shapes of each pyramid
        level, so the shape models are scaled.

        If ``False``, the reference frames of all levels are the mean shape of
        the highest level, so the shape models are not scaled; they have the
        same size.

        Note that from our experience, if ``scaled_shape_models`` is ``False``,
        AAMs tend to have slightly better performance.

    pyramid_on_features : `boolean`, optional
        If ``True``, the feature space was computed once at the highest scale and
        the Gaussian pyramid was applied on the feature images.

        If ``False``, the Gaussian pyramid was applied on the original images
        (intensities) and then features were extracted at each level.

        Note that from our experience, if ``pyramid_on_features`` is ``True``,
        AAMs tend to have slightly better performance.
    """
    def __init__(self, shape_models, appearance_models, n_training_images,
                 patch_shape, transform, features, reference_shape,
                 downscale, scaled_shape_models, pyramid_on_features):
        super(PatchBasedAAM, self).__init__(
            shape_models, appearance_models, n_training_images, transform,
            features, reference_shape, downscale, scaled_shape_models,
            pyramid_on_features)
        self.patch_shape = patch_shape

    def _build_reference_frame(self, reference_shape, landmarks):
        return build_patch_reference_frame(
            reference_shape, patch_shape=self.patch_shape)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'Patch-Based Active Appearance Model'

    def __str__(self):
        out = super(PatchBasedAAM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        out_splitted.insert(5, "   - Patch size is {}W x {}H.".format(
            self.patch_shape[1], self.patch_shape[0]))
        return '\n'.join(out_splitted)
