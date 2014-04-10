from __future__ import division

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
        return self.clm.reference_shape

    @property
    def feature_type(self):
        return self.clm.feature_type

    @property
    def n_levels(self):
        return self.clm.n_levels

    @property
    def downscale(self):
        return self.clm.downscale

    @property
    def scaled_levels(self):
        return self.clm.scaled_levels

    @property
    def interpolator(self):
        return self.clm.interpolator

    # TODO: Can this be moved up?
    def _prepare_image(self, image, initial_shape, gt_shape=None):
        r"""
        The image is first rescaled wrt the reference_landmarks, then
        smoothing or gaussian pyramid are computed and, finally, features
        are extracted from each pyramidal element.
        """
        image.landmarks['initial_shape'] = initial_shape
        image = image.rescale_to_reference_shape(
            self.reference_shape, group='initial_shape',
            interpolator=self.interpolator)

        if gt_shape:
            image.landmarks['gt_shape'] = initial_shape

        if self.n_levels > 1:
            if self.scaled_levels:
                pyramid = image.gaussian_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)
            else:
                pyramid = image.smoothing_pyramid(
                    n_levels=self.n_levels, downscale=self.downscale)
            images = [compute_features(i, self.feature_type)
                      for i in pyramid]
            images.reverse()
        else:
            images = [compute_features(image, self.feature_type)]

        return images


# TODO: document me
# TODO: Residuals (SSD) is not used at the moment
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

    residual: :class:`menpo.fit.gradientdescent.residual`, optional
        The residual class to be used

        Default: 'SSD'

    pdm: :class:`menpo.transform.ModelDrivenTransform`, optional
        The point distribution transform class to be used.

        Default: OrthoPDMTransform

    global_transform: :class:`menpo.transform.affine`, optional
        The global transform class to be used by the previous
        md_transform_cls. Currently, only
        :class:`menpo.transform.affine.Similarity` is supported.

        Default: Similarity

    n_shape: list, optional
        The number of shape components to be used per fitting level.
        If None, for each shape model n_active_components will be used.

        Default: None

    n_appearance: list, optional
        The number of appearance components to be used per fitting level.
        If None, for each appearance model n_active_components will be used.

        Default: None
    """
    def __init__(self, clm, algorithm=RegularizedLandmarkMeanShift,
                 residual=SSD, pdm_transform=OrthoPDM,
                 global_transform=AlignmentSimilarity, n_shape=None):
        super(GradientDescentCLMFitter, self).__init__(clm)
        self._set_up(algorithm=algorithm, residual=residual,
                     pdm_transform=pdm_transform,
                     global_transform=global_transform,
                     n_shape=n_shape)

    @property
    def algorithm(self):
        return 'GD-CLM-' + self._fitters[0].algorithm

    # TODO: document me
    def _set_up(self, algorithm=RegularizedLandmarkMeanShift, residual=SSD,
                pdm_transform=OrthoPDM,
                global_transform=AlignmentSimilarity, n_shape=None):
        r"""
        Sets up the gradient descent fitter object.

        Parameters
        -----------
        clm: :class:`menpo.fitmultilevel.clm.builder.CLM`
            The Constrained Local Model to be use.

        algorithm: :class:`menpo.fit.gradientdescent.base`, optional
            The Gradient Descent class to be used.

            Default: RegularizedLandmarkMeanShift

        residual: :class:`menpo.fit.gradientdescent.residual`, optional
            The residual class to be used

            Default: 'SSD'

        pdm: :class:`menpo.transform.ModelDrivenTransform`, optional
            The point distribution transform class to be used.

            Default: OrthoPDMTransform

        global_transform: :class:`menpo.transform.affine`, optional
            The global transform class to be used by the previous
            md_transform_cls. Currently, only
            :class:`menpo.transform.affine.Similarity` is supported.

            Default: Similarity

        n_shape: list, optional
            The number of shape components to be used per fitting level.
            If None, for each shape model n_active_components will be used.

            Default: None

        n_appearance: list, optional
            The number of appearance components to be used per fitting level.
            If None, for each appearance model n_active_components will be used.

            Default: None
        """
        if n_shape is None:
            n_shape = [sm.n_active_components
                       for sm in self.clm.shape_models]
        if type(n_shape) is int:
            n_shape = [n_shape for _ in range(self.clm.n_levels)]
        elif len(n_shape) is 1 and self.clm.n_levels > 1:
            n_shape = [n_shape[0] for _ in range(self.clm.n_levels)]
        elif len(n_shape) is not self.clm.n_levels:
            raise ValueError('n_shape can be integer, integer list '
                             'containing 1 or {} elements or '
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

            self._fitters.append(algorithm(clf,
                                           self.clm.patch_shape,
                                           pdm_trans))
