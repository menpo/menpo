from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.featurefunctions import compute_features
from menpo.fitmultilevel.aam.base import AAMFitter
from menpo.fitmultilevel.clm.base import CLMFitter


# TODO: document me
class SupervisedDescentFitter(MultilevelFitter):
    r"""
    """
    def _set_up(self):
        pass

    def fit(self, image, group=None, label='all',
            initialization='from_gt_shape', noise_std=0.0, rotation=False,
            max_iters=None, verbose=True, view=False, error_type='me_norm',
            **kwargs):
        if max_iters is None:
            max_iters = self.n_levels
        return super(SupervisedDescentFitter, self).fit(
            image, group=group, label=label, initialization=initialization,
            noise_std=noise_std, rotation=rotation, max_iters=max_iters,
            verbose=verbose, view=view, error_type=error_type,
            **kwargs)

    def fit_images(self, images, group=None, label='all',
                   initialization='from_gt_shape', noise_std=0.0,
                   rotation=False, max_iters=None, verbose=True, view=False,
                   error_type='me_norm', **kwargs):
        if max_iters is None:
            max_iters = self.n_levels
        return super(SupervisedDescentFitter, self).fit_images(
            images, group=group, label=label, initialization=initialization,
            noise_std=noise_std, rotation=rotation, max_iters=max_iters,
            verbose=verbose, view=view, error_type=error_type,
            **kwargs)


#TODO: Document me
class SupervisedDescentMethodFitter(SupervisedDescentFitter):
    r"""
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
            images = [compute_features(i, self.feature_type)
                      for i in pyramid]
            images.reverse()
        else:
            images = [compute_features(image, self.feature_type)]

        return images


#TODO: Document me
class SupervisedDescentAAMFitter(AAMFitter, SupervisedDescentFitter):
    r"""
    """
    def __init__(self, aam, regressors):
        super(SupervisedDescentAAMFitter, self).__init__(aam)
        self._fitters = regressors

    @property
    def algorithm(self):
        return 'SD-AAM' + self._fitters[0].algorithm


#TODO: document me
class SupervisedDescentCLMFitter(CLMFitter, SupervisedDescentFitter):
    r"""
    """
    def __init__(self, clm, regressors):
        super(SupervisedDescentCLMFitter, self).__init__(clm)
        self._fitters = regressors

    @property
    def algorithm(self):
        return 'SD-CLM' + self._fitters[0].algorithm
