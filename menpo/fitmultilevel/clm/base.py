from __future__ import division
import numpy as np

from menpo.image import Image
from menpo.transform import AlignmentSimilarity
from menpo.model.pdm import PDM, OrthoPDM
from menpo.fit.gradientdescent import RegularizedLandmarkMeanShift
from menpo.fit.gradientdescent.residual import SSD
from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.featurefunctions import compute_features


class CLMFitter(MultilevelFitter):
    r"""
    Mixin for Constrained Local Models Fitters.

    Parameters
    -----------
    clm: :class:`menpo.fitmultilevel.clm.builder.CLM`
        The Constrained Local Model to be used.
    """
    def __init__(self, clm):
        self.clm = clm

    @property
    def reference_shape(self):
        r"""
        The reference shape of the trained CLM.

        : `menpo.shape.Pointcloud`
        """
        return self.clm.reference_shape

    @property
    def feature_type(self):
        r"""
        The feature type per pyramid level of the trained CLM. Note that they
        are stored from lowest to highest level resolution.

        : list
        """
        return self.clm.feature_type

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during building the CLM.

        : int
        """
        return self.clm.n_levels

    @property
    def downscale(self):
        r"""
        The downscale per pyramidal level used during building the CLM.
        The scale factor is: (downscale ** k) for k in range(n_levels)

        : float
        """
        return self.clm.downscale

    @property
    def pyramid_on_features(self):
        r"""
        Flag that controls the Gaussian pyramid of the testing image based on
        the pyramid used during building the CLM.
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

        : boolean
        """
        return self.clm.pyramid_on_features

    @property
    def interpolator(self):
        r"""
        The interpolator used during training.

        : str
        """
        return self.clm.interpolator


class GradientDescentCLMFitter(CLMFitter):
    r"""
    Gradient Descent based Fitter for Constrained Local Models.

    Parameters
    -----------
    clm: :class:`menpo.fitmultilevel.clm.builder.CLM`
        The Constrained Local Model to be used.
    algorithm: :class:`menpo.fit.gradientdescent.base`, optional
        The Gradient Descent class to be used.

        Default: RegularizedLandmarkMeanShift
    pdm_transform: :class:`menpo.transform.ModelDrivenTransform`, optional
        The point distribution transform class to be used.

        Default: OrthoPDMTransform
    global_transform: :class:`menpo.transform.affine`, optional
        The global transform class to be used by the previous
        md_transform_cls. Currently, only
        :class:`menpo.transform.affine.AlignmentSimilarity` is supported.

        Default: AlignmentSimilarity
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
    """
    def __init__(self, clm, algorithm=RegularizedLandmarkMeanShift,
                 pdm_transform=OrthoPDM, global_transform=AlignmentSimilarity,
                 n_shape=None):
        super(GradientDescentCLMFitter, self).__init__(clm)
        # TODO: Add residual as parameter, when residuals are properly defined
        residual = SSD
        self._set_up(algorithm=algorithm, residual=residual,
                     pdm_transform=pdm_transform,
                     global_transform=global_transform, n_shape=n_shape)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the Gradient
        Descent family.

        : str
        """
        return 'GD-CLM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=RegularizedLandmarkMeanShift, residual=SSD,
                pdm_transform=OrthoPDM, global_transform=AlignmentSimilarity,
                n_shape=None):
        r"""
        Sets up the gradient descent fitter object.

        Parameters
        -----------
        algorithm: :class:`menpo.fit.gradientdescent.base`, optional
            The Gradient Descent class to be used.

            Default: RegularizedLandmarkMeanShift
        residual: :class:`menpo.fit.gradientdescent.residual`, optional
            The residual class to be used

            Default: 'SSD'
        pdm_transform: :class:`menpo.transform.ModelDrivenTransform`, optional
            The point distribution transform class to be used.

            Default: OrthoPDMTransform
        global_transform: :class:`menpo.transform.affine`, optional
            The global transform class to be used by the previous
            md_transform_cls. Currently, only
            :class:`menpo.transform.affine.AlignmentSimilarity` is supported.

            Default: AlignmentSimilarity
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
        """
        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) is 1 and self.clm.n_levels > 1:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) is self.clm.n_levels:
                for sm, n in zip(self.clm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float, '
                                 'an integer or float list containing 1 '
                                 'or {} elements or else '
                                 'None'.format(self.clm.n_levels))

        self._fitters = []
        for j, (sm, clf) in enumerate(zip(self.clm.shape_models,
                                          self.clm.classifiers)):
            if n_shape is not None:
                sm.n_active_components = n_shape[j]
            if pdm_transform is not PDM:
                pdm_trans = pdm_transform(sm, global_transform)
            else:
                pdm_trans = pdm_transform(sm)
            self._fitters.append(algorithm(clf, self.clm.patch_shape,
                                           pdm_trans))

    def __str__(self):
        out = "{0} Fitter\n" \
              " - Gradient-Descent {1}\n" \
              " - Transform is {2}.\n" \
              " - {3} training images.\n".format(
              self.clm._str_title, self._fitters[0].algorithm,
              self._fitters[0].transform.__class__.__name__,
              self.clm.n_training_images)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        temp_img = Image(image_data=np.random.rand(50, 50))
        for j in range(self.n_levels):
            rj = self.n_levels - j - 1
            temp = compute_features(temp_img, self.feature_type[rj])
            n_channels.append(temp.n_channels)
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
            if self.clm.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.clm.patch_shape[1],
                                      self.clm.patch_shape[0])

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.clm.patch_shape[1],
                                      self.clm.patch_shape[0])
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str)
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if self.pyramid_on_features is False:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                out = "{0}     - {1} motion components\n     - {2} {3} " \
                      "classifiers.\n".format(
                      out, self._fitters[i].transform.n_parameters,
                      len(self._fitters[i].classifiers),
                      self._fitters[i].classifiers[0].func_name)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - {4} motion components\n   - {5} {6} " \
                  "classifiers.".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  out, self._fitters[0].transform.n_parameters,
                  len(self._fitters[0].classifiers),
                  self._fitters[0].classifiers[0].func_name)
        return out
