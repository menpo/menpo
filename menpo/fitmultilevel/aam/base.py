from __future__ import division

from menpo.transform import AlignmentSimilarity
from menpo.transform.modeldriven import OrthoMDTransform, ModelDrivenTransform
from menpo.fit.lucaskanade.residual import LSIntensity
from menpo.fit.lucaskanade.appearance import AlternatingInverseCompositional
from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.fittingresult import AAMMultilevelFittingResult
from menpo.fitmultilevel.featurefunctions import compute_features


class AAMFitter(MultilevelFitter):
    r"""
    Mixin for Active Appearance Models Fitters.

    Parameters
    -----------
    aam: :class:`menpo.fitmultilevel.aam.builder.AAM`
        The Active Appearance Model to be used.
    """
    def __init__(self, aam):
        self.aam = aam

    @property
    def reference_shape(self):
        r"""
        The reference shape of the trained AAM.

        : `menpo.shape.Pointcloud`
        """
        return self.aam.reference_shape

    @property
    def feature_type(self):
        r"""
        The feature type per pyramid level of the trained AAM. Note that they
        are stored from lowest to highest level resolution.

        : list
        """
        return self.aam.feature_type

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during building the AAM.

        : int
        """
        return self.aam.n_levels

    @property
    def downscale(self):
        r"""
        The downscale per pyramidal level used during building the AAM.
        The scale factor is: (downscale ** k) for k in range(n_levels)

        : float
        """
        return self.aam.downscale

    @property
    def pyramid_on_features(self):
        r"""
        Flag that controls the Gaussian pyramid of the testing image based on
        the pyramid used during building the AAM.
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

        : boolean
        """
        return self.aam.pyramid_on_features

    @property
    def interpolator(self):
        r"""
        The interpolator used for warping.

        : str
        """
        return self.aam.interpolator

    # TODO: Can this be moved up?
    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        The image is first rescaled wrt the reference_landmarks and then the
        gaussian pyramid is computed. Depending on the pyramid_on_features
        flag, the pyramid is either applied on the feature image or
        features are extracted at each pyramidal level.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The image to be fitted.
        initial_shape: class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The original ground truth shape associated to the image.

            Default: None

        Returns
        -------
        images: list of :class:`menpo.image.masked.MaskedImage`
            List of images, each being the result of applying the pyramid.
        """
        # rescale image wrt the scale factor between reference_shape and
        # initial_shape
        image.landmarks['initial_shape'] = initial_shape
        image = image.rescale_to_reference_shape(
            self.reference_shape, group='initial_shape',
            interpolator=self.interpolator)

        # attach given ground truth shape
        if gt_shape:
            image.landmarks['gt_shape'] = gt_shape

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
        return images

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None, error_type='me_norm'):
        r"""
        Creates the :class: `menpo.aam.fitting.MultipleFitting` object
        associated with a particular Fitter object.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The original image to be fitted.
        fitting_results: :class:`menpo.fitmultilevel.FittingResult` list
            A list of basic fitting objects containing the state of the
            different fitting levels.
        affine_correction: :class: `menpo.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        fitting: :class:`menpo.aam.Fitting`
            The fitting object that will hold the state of the fitter.
        """
        return AAMMultilevelFittingResult(
            image, self, fitting_results, affine_correction, gt_shape=gt_shape,
            error_type=error_type)


class LucasKanadeAAMFitter(AAMFitter):
    r"""
    Lucas-Kanade based Fitter for Active Appearance Models.

    Parameters
    -----------
    aam: :class:`menpo.fitmultilevel.aam.builder.AAM`
        The Active Appearance Model to be used.
    algorithm: :class:`menpo.fit.lucaskanade.appearance`, optional
        The Lucas-Kanade class to be used.

        Default: AlternatingInverseCompositional
    md_transform: :class:`menpo.transform.ModelDrivenTransform`,
                      optional
        The model driven transform class to be used.

        Default: OrthoMDTransform
    global_transform: :class:`menpo.transform.affine`, optional
        The global transform class to be used by the previous
        md_transform_cls. Currently, only
        :class:`menpo.transform.affine.AlignmentSimilarity` is supported.

        Default: AlignmentSimilarity
    n_shape: list, optional
        The number of shape components to be used per fitting level.
        If None, for each shape model n_active_components will be used.

        Default: None
    n_appearance: list, optional
        The number of appearance components to be used per fitting level.
        If None, for each appearance model n_active_components will be used.

        Default: None
    """
    def __init__(self, aam, algorithm=AlternatingInverseCompositional,
                 md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None):
        super(LucasKanadeAAMFitter, self).__init__(aam)
        # TODO: Add residual as parameter, when residuals are properly defined
        residual = LSIntensity
        self._set_up(algorithm=algorithm, residual=residual,
                     md_transform=md_transform,
                     global_transform=global_transform,
                     n_shape=n_shape, n_appearance=n_appearance)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the Lucas-Kanade
        family.

        : str
        """
        return 'LK-AAM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=AlternatingInverseCompositional,
                residual=LSIntensity, md_transform=OrthoMDTransform,
                global_transform=AlignmentSimilarity, n_shape=None,
                n_appearance=None):
        r"""
        Sets up the Lucas-Kanade fitter object.

        Parameters
        -----------
        algorithm: :class:`menpo.lucaskanade.appearance`, optional
            The Lucas-Kanade class to be used.

            Default: AlternatingInverseCompositional
        residual: :class:`menpo.lucaskanade.residual`, optional
            The residual class to be used

            Default: 'LSIntensity'
        md_transform: :class:`menpo.transform.ModelDrivenTransform`,
                          optional
            The model driven transform class to be used.

            Default: OrthoMDTransform
        global_trans: :class:`menpo.transform.affine`, optional
            The global transform class to be used by the previous
            md_transform. Currently, only
            :class:`menpo.transform.affine.Similarity` is supported.

            Default: Similarity
        n_shape: int > 1 or 0. <= float <= 1. or None, or a list of those,
                 optional
            The number of shape components to be used per fitting level.

            If list of length n_levels, then a number of components is defined
            per level. The first element of the list corresponds to the lowest
            pyramidal level and so on.

            If not a list or a list with length 1, then the specified number of
            components will be used for all levels.

            Per level:
            If None, all the available shape components (n_active_componenets)
            will be used.
            If int > 1, a specific number of shape components is specified.
            If 0. <= float <= 1., it specifies the variance percentage that is
            captured by the components.

            Default: None
        n_appearance: int > 1 or 0. <= float <= 1. or None, or a list of those,
                      optional
            The number of appearance components to be used per fitting level.

            If list of length n_levels, then a number of components is defined
            per level. The first element of the list corresponds to the lowest
            pyramidal level and so on.

            If not a list or a list with length 1, then the specified number of
            components will be used for all levels.

            Per level:
            If None, all the available appearance components
            (n_active_componenets) will be used.
            If int > 1, a specific number of appearance components is specified
            If 0. <= float <= 1., it specifies the variance percentage that is
            captured by the components.

            Default: None
        """
        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) is 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) is self.aam.n_levels:
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
            elif len(n_appearance) is 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) is self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.aam.n_levels))

        self._fitters = []
        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):

            if md_transform is not ModelDrivenTransform:
                md_trans = md_transform(
                    sm, self.aam.transform, global_transform,
                    source=am.mean.landmarks['source'].lms)
            else:
                md_trans = md_transform(
                    sm, self.aam.transform,
                    source=am.mean.landmarks['source'].lms)
            self._fitters.append(algorithm(am, residual(), md_trans))

    def __str__(self):
        out = "{0} Fitter\n" \
              " - Lucas-Kanade {1}\n" \
              " - Transform is {2} and residual is {3}.\n" \
              " - {4} training images.\n".format(
              self.aam._str_title, self._fitters[0].algorithm,
              self._fitters[0].transform.__class__.__name__,
              self._fitters[0].residual.type, self.aam.n_training_images)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        for j in range(self.n_levels):
            n_channels.append(
                self._fitters[j].appearance_model.template_instance.n_channels)
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
        if self.n_levels > 1:
            if self.aam.scaled_shape_models:
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
                if self.aam.scaled_shape_models is False:
                    out = "{}   - Reference frames of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out, self._fitters[0].appearance_model.n_features,
                          self._fitters[0].template.n_true_pixels,
                          n_channels[0], self._fitters[0].template._str_shape,
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
                if (self.aam.scaled_shape_models or
                        self.pyramid_on_features is False):
                    out = "{}     - Reference frame of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out, self._fitters[i].appearance_model.n_features,
                          self._fitters[i].template.n_true_pixels,
                          n_channels[i], self._fitters[i].template._str_shape,
                          n_channels[i])
                out = "{0}     - {1} motion components\n     - {2} active " \
                      "appearance components ({3:.2f}% of original " \
                      "variance)\n".format(
                      out, self._fitters[i].transform.n_parameters,
                      self._fitters[i].appearance_model.n_active_components,
                      self._fitters[i].appearance_model.variance_ratio * 100)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - Reference frame of length {4} ({5} x {6}C, " \
                  "{7} x {8}C)\n   - {9} motion parameters\n" \
                  "   - {10} appearance components ({11:.2f}% of original " \
                  "variance)\n".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  self._fitters[0].appearance_model.n_features,
                  self._fitters[0].template.n_true_pixels,
                  n_channels[0], self._fitters[0].template._str_shape,
                  n_channels[0], self._fitters[0].transform.n_parameters,
                  self._fitters[0].appearance_model.n_active_components,
                  self._fitters[0].appearance_model.variance_ratio * 100)
        return out
