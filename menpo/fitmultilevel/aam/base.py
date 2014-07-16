from __future__ import division

from menpo.transform import AlignmentSimilarity
from menpo.transform.modeldriven import OrthoMDTransform, ModelDrivenTransform
from menpo.fit.lucaskanade.residual import LSIntensity
from menpo.fit.lucaskanade.appearance import AlternatingInverseCompositional
from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.fittingresult import AAMMultilevelFittingResult


class AAMFitter(MultilevelFitter):
    r"""
    Abstract Interface for defining Active Appearance Models Fitters.

    Parameters
    -----------
    aam : :map:`AAM`
        The Active Appearance Model to be used.
    """
    def __init__(self, aam):
        self.aam = aam

    @property
    def reference_shape(self):
        r"""
        The reference shape of the AAM.

        :type: :map:`PointCloud`
        """
        return self.aam.reference_shape

    @property
    def feature_type(self):
        r"""
        The feature extracted at each pyramidal level during AAM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self.aam.feature_type

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during AAM building.

        :type: `int`
        """
        return self.aam.n_levels

    @property
    def downscale(self):
        r"""
        The downscale used to generate the final scale factor applied at
        each pyramidal level during AAM building.
        The scale factor is computed as:

            ``(downscale ** k) for k in range(n_levels)``

        :type: `float`
        """
        return self.aam.downscale

    @property
    def pyramid_on_features(self):
        r"""
        Flag that defined the nature of Gaussian pyramid used to build the
        AAM.
        If ``True``, the feature space is computed once at the highest scale
        and the Gaussian pyramid is applied to the feature images.
        If ``False``, the Gaussian pyramid is applied to the original images
        and features are extracted at each level.

        :type: `boolean`
        """
        return self.aam.pyramid_on_features

    @property
    def interpolator(self):
        r"""
        The interpolator used during AAM building.

        :type: `str`
        """
        return self.aam.interpolator

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None, error_type='me_norm'):
        r"""
        Creates a :map:`AAMMultilevelFittingResult` associated to a
        particular fitting of the AAM fitter.

        Parameters
        -----------
        image : :map:`Image` or subclass
            The image to be fitted.

        fitting_results : `list` of :map:`FittingResult`
            A list of fitting result objects containing the state of the
            the fitting for each pyramidal level.

        affine_correction : :map:`Affine`
            An affine transform that maps the result of the top resolution
            level to the scale space of the original image.

        gt_shape : :map:`PointCloud`, optional
            The ground truth shape associated to the image.

        error_type : 'me_norm', 'me' or 'rmse', optional
            Specifies how the error between the fitted and ground truth
            shapes must be computed.

        Returns
        -------
        fitting : :map:`AAMMultilevelFittingResult`
            A fitting result object that will hold the state of the AAM
            fitter for a particular fitting.
        """
        return AAMMultilevelFittingResult(
            image, self, fitting_results, affine_correction, gt_shape=gt_shape,
            error_type=error_type)


class LucasKanadeAAMFitter(AAMFitter):
    r"""
    Lucas-Kanade based :map:`Fitter` for Active Appearance Models.

    Parameters
    -----------
    aam : :map:`AAM`
        The Active Appearance Model to be used.
    algorithm : subclass of :map:`AppearanceLucasKanade`, optional
        The Appearance Lucas-Kanade class to be used.

    md_transform : :map:`ModelDrivenTransform` or subclass, optional
        The model driven transform class to be used.

    global_transform : subclass of :map:`HomogFamilyAlignment`, optional
        The global transform class to be used by the previous pdm.

        .. note::

            Only :map:`AlignmentSimilarity` is supported when
            ``pdm_transform`` is set to :map:`AlignmentSimilarity`.

            ``global_transform`` has no effect when ``md_transform`` is
            specifically set to map:`MDTransform`

    n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
        previous or ``None``, optional
        The number of shape components or amount of shape variance to be
        used per pyramidal level.

        If `None`, all available shape components ``(n_active_components)``
        will be used.
        If `int` ``> 1``, the specified number of shape components will be
        used.
        If ``0. <=`` `float` ``<= 1.``, the number of components capturing the
        specified variance ratio will be computed and used.

        If `list` of length ``n_levels``, then the number of components is
        defined per level. The first element of the list corresponds to the
        lowest pyramidal level and so on.
        If not a `list` or a `list` of length 1, then the specified number of
        components will be used for all levels.

    n_appearance : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
        previous or ``None``, optional
        The number of appearance components or amount of appearance variance
        to be used per pyramidal level.

        If `None`, all available appearance components
        ``(n_active_components)`` will be used.
        If `int` ``> 1``, the specified number of appearance components will
        be used.
        If ``0. <=`` `float` ``<= 1.``, the number of appearance components
        capturing the specified variance ratio will be computed and used.

        If `list` of length ``n_levels``, then the number of components is
        defined per level. The first element of the list corresponds to the
        lowest pyramidal level and so on.
        If not a `list` or a `list` of length 1, then the specified number of
        components will be used for all levels.
    """
    def __init__(self, aam, algorithm=AlternatingInverseCompositional,
                 md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None, **kwargs):
        super(LucasKanadeAAMFitter, self).__init__(aam)
        # TODO: Add residual as parameter, when residuals are properly defined
        residual = LSIntensity
        self._set_up(algorithm=algorithm, residual=residual,
                     md_transform=md_transform,
                     global_transform=global_transform, n_shape=n_shape,
                     n_appearance=n_appearance, **kwargs)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the name of fitting algorithm.

        :type: `str`
        """
        return 'LK-AAM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=AlternatingInverseCompositional,
                residual=LSIntensity, md_transform=OrthoMDTransform,
                global_transform=AlignmentSimilarity, n_shape=None,
                n_appearance=None, **kwargs):
        r"""
        Sets up the Lucas-Kanade fitter object.

        Parameters
        -----------
        algorithm : subclass of :map:`AppearanceLucasKanade`, optional
            The Appearance Lucas-Kanade class to be used.

        md_transform : :map:`ModelDrivenTransform` or subclass, optional
            The model driven transform class to be used.

        global_transform : subclass of :map:`HomogFamilyAlignment`, optional
            The global transform class to be used by the previous pdm.

            .. note::

                Only :map:`AlignmentSimilarity` is supported when
                ``pdm_transform`` is set to :map:`AlignmentSimilarity`.

                ``global_transform`` has no effect when ``md_transform`` is
                specifically set to map:`MDTransform`

        n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
            previous or ``None``, optional
            The number of shape components or amount of shape variance to be
            used per pyramidal level.

            If `None`, all available shape components ``(n_active_components)``
            will be used.
            If `int` ``> 1``, the specified number of shape components will be
            used.
            If ``0. <=`` `float` ``<= 1.``, the number of components capturing the
            specified variance ratio will be computed and used.

            If `list` of length ``n_levels``, then the number of components is
            defined per level. The first element of the list corresponds to the
            lowest pyramidal level and so on.
            If not a `list` or a `list` of length 1, then the specified number of
            components will be used for all levels.

        n_appearance : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
            previous or ``None``, optional
            The number of appearance components or amount of appearance variance
            to be used per pyramidal level.

            If `None`, all available appearance components
            ``(n_active_components)`` will be used.
            If `int` ``> 1``, the specified number of appearance components will
            be used.
            If ``0. <=`` `float` ``<= 1.``, the number of appearance components
            capturing the specified variance ratio will be computed and used.

            If `list` of length ``n_levels``, then the number of components is
            defined per level. The first element of the list corresponds to the
            lowest pyramidal level and so on.
            If not a `list` or a `list` of length 1, then the specified number of
            components will be used for all levels.

        Raises
        -------
        ValueError
            ``n_shape`` can be an `int`, `float`, ``None`` or a `list`
            containing ``1`` or ``n_levels`` of those.
        ValueError
            ``n_appearance`` can be an `int`, `float`, `None` or a `list`
            containing ``1`` or ``n_levels`` of those.
        """
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
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

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
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

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
            self._fitters.append(algorithm(am, residual(), md_trans,
                                           **kwargs))

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
                                                    n_channels[0], ch_str[0])
                if not self.aam.scaled_shape_models:
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
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                if (self.aam.scaled_shape_models or
                        (not self.pyramid_on_features)):
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
