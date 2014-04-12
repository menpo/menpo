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
    aam: :class:`menpo.aam.AAM`
        The Active Appearance Model to be used.
    """
    def __init__(self, aam):
        self.aam = aam

    @property
    def reference_shape(self):
        return self.aam.reference_shape

    @property
    def feature_type(self):
        return self.aam.feature_type

    @property
    def n_levels(self):
        return self.aam.n_levels

    @property
    def downscale(self):
        return self.aam.downscale

    @property
    def scaled_levels(self):
        return self.aam.scaled_levels

    @property
    def interpolator(self):
        return self.aam.interpolator

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

    def _create_fitting(self, image, fittings, affine_correction,
                        gt_shape=None, error_type='me_norm'):
        return AAMMultilevelFittingResult(
            image, self, fittings, affine_correction, gt_shape=gt_shape,
            error_type=error_type)


class LucasKanadeAAMFitter(AAMFitter):
    r"""
    Lucas-Kanade based Fitter for Active Appearance Models.

    Parameters
    -----------
    aam: :class:`menpo.fitmultilevel.aam.builder.AAM`
        The Active Appearance Model to be use.

    algorithm: :class:`menpo.fit.lucaskanade.appearance`, optional
        The Lucas-Kanade class to be used.

        Default: AlternatingInverseCompositional

    residual: :class:`menpo.fit.lucaskanade.residual`, optional
        The residual class to be used

        Default: 'LSIntensity'

    md_transform: :class:`menpo.transform.ModelDrivenTransform`,
                      optional
        The model driven transform class to be used.

        Default: OrthoMDTransform

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
    def __init__(self, aam, algorithm=AlternatingInverseCompositional,
                 residual=LSIntensity, md_transform=OrthoMDTransform,
                 global_transform=AlignmentSimilarity, n_shape=None,
                 n_appearance=None):
        super(LucasKanadeAAMFitter, self).__init__(aam)
        self._set_up(algorithm=algorithm, residual=residual,
                     md_transform=md_transform,
                     global_transform=global_transform,
                     n_shape=n_shape, n_appearance=n_appearance)

    @property
    def algorithm(self):
        return 'LK-AAM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=AlternatingInverseCompositional,
                residual=LSIntensity, md_transform=OrthoMDTransform,
                global_transform=AlignmentSimilarity, n_shape=None,
                n_appearance=None):
        r"""
        Sets up the lucas-kanade fitter object.

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

        n_shape: list, optional
            The number of shape components to be used per fitting level.
            If None, for each shape model n_active_components will be used.

            Default: None

        n_appearance: list, optional
            The number of appearance components to be used per fitting level.
            If None, for each appearance model n_active_components will be used.

            Default: None
        """
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
