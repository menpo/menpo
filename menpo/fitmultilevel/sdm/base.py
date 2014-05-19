from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.featurefunctions import compute_features
from menpo.fitmultilevel.aam.base import AAMFitter
from menpo.fitmultilevel.clm.base import CLMFitter


class SDFitter(MultilevelFitter):
    r"""
    Mixin for Supervised Descent Fitters.
    """
    def _set_up(self):
        pass

    def fit(self, image, initial_shape, max_iters=None, gt_shape=None,
            error_type='me_norm', verbose=False, view=False, **kwargs):
        if max_iters is None:
            max_iters = self.n_levels
        return MultilevelFitter.fit(self, image, initial_shape,
                                    max_iters=max_iters, gt_shape=gt_shape,
                                    error_type=error_type, verbose=verbose,
                                    view=view, **kwargs)


#TODO: Document me
class SDMFitter(SDFitter):
    r"""
    Supervised Descent Method.

    References
    ----------
    Xuehan Xiong and Fernando De la Torre Fernando, "Supervised Descent Method
    and its Applications to Face Alignment," IEEE International Conference on
    Computer Vision and Pattern Recognition (CVPR), May, 2013.

    Parameters
    -----------
    regressors:

    feature_type: :class:`menpo.fit.gradientdescent.base`, optional
        The Gradient Descent class to be used.

    reference_shape: PointCloud
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale: float
        The constant downscale factor used to create the different levels of
        training levels. For example, a factor of 2 would imply that the
        second level of the SDM pyramid is half the width and half the
        height of the first. The third would be 1/2 * 1/2 = 1/4 the width
        and 1/4 the height of the original.

    scaled_levels: boolean
        Boolean value specifying whether the levels are scaled or not.

    interpolator: string
        The interpolator that was used during training.
    """
    def __init__(self, regressors, feature_type, reference_shape, downscale,
                 scaled_levels, interpolator):
        self._fitters = regressors
        self._feature_type = feature_type
        self._reference_shape = reference_shape
        self._downscale = downscale
        self._scaled_levels = scaled_levels
        self._interpolator = interpolator


    @property
    def algorithm(self):
        return 'SDM-' + self._fitters[0].algorithm

    @property
    def reference_shape(self):
        return self._reference_shape

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def n_levels(self):
        return len(self._fitters)

    @property
    def downscale(self):
        return self._downscale

    @property
    def scaled_levels(self):
        return self._scaled_levels

    @property
    def interpolator(self):
        return self._interpolator

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
            images = [compute_features(i, self.feature_type[j])
                      for j, i in enumerate(pyramid)]
            images.reverse()
        else:
            images = [compute_features(image, self.feature_type[0])]

        return images


#TODO: Document me
class SDAAMFitter(AAMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for AAMs

    Parameters
    -----------
    aam: :class:`menpo.fitmultilevel.aam.builder.AAM`
        The Active Appearance Model to be used.

    regressors:
    """
    def __init__(self, aam, regressors):
        super(SDAAMFitter, self).__init__(aam)
        self._fitters = regressors

    @property
    def algorithm(self):
        return 'SD-AAM' + self._fitters[0].algorithm


#TODO: document me
class SDCLMFitter(CLMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for CLMs

    Parameters
    -----------
    clm: :class:`menpo.fitmultilevel.clm.builder.CLM`
        The Constrained Local Model to be used.

    regressors:

    References
    ----------
    A. Asthana, S. Zafeiriou, S. Cheng, M. Pantic. 2013 IEEE Conference on
    Computer Vision and Pattern Recognition (CVPR 2013). Portland, Oregon,
    USA, June 2013.
    """
    def __init__(self, clm, regressors):
        super(SDCLMFitter, self).__init__(clm)
        self._fitters = regressors

    @property
    def algorithm(self):
        return 'SD-CLM' + self._fitters[0].algorithm
