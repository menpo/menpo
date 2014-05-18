from menpo.fitmultilevel.base import MultilevelFitter
from menpo.fitmultilevel.featurefunctions import compute_features
from menpo.fitmultilevel.aam.base import AAMFitter
from menpo.fitmultilevel.clm.base import CLMFitter


class SDFitter(MultilevelFitter):
    r"""
    Mixin for Supervised Descent Fitters.
    """
    def _set_up(self):
        r"""
        Sets up the SD fitter object.
        """
        pass

    def fit(self, image, initial_shape, max_iters=None, gt_shape=None,
            error_type='me_norm', verbose=False, view=False, **kwargs):
        r"""
        Fits a single image.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The image to be fitted.
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50
        gt_shape: PointCloud
            The groundtruth shape of the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional.
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'
        verbose: boolean, optional
            Whether or not to print information related to the fitting
            results (such as: final error, convergence, ...).

            Default: True
        view: boolean, optional
            Whether or not the fitting results are to be displayed.

            Default: False

        **kwargs:

        Returns
        -------
        FittingList: :class:`menpo.fitmultilevel.fittingresult`
            A fitting result object.
        """
        if max_iters is None:
            max_iters = self.n_levels
        return MultilevelFitter.fit(self, image, initial_shape,
                                    max_iters=max_iters, gt_shape=gt_shape,
                                    error_type=error_type, verbose=verbose,
                                    view=view, **kwargs)


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
    regressors: :class: menpo.fit.regression.RegressorTrainer
        The trained regressors.
    feature_type: None or string or function/closure or list of those, Optional
        If list of length n_levels, then a feature is defined per level.
        However, this requires that the pyramid_on_features flag is disabled,
        so that the features are extracted at each level. The first element of
        the list specifies the features to be extracted at the lowest pyramidal
        level and so on.

        If not a list or a list with length 1, then:
            If pyramid_on_features is True, the specified feature will be
            applied to the highest level.
            If pyramid_on_features is False, the specified feature will be
            applied to all pyramid levels.

        Per level:
        If None, the appearance model will be built using the original image
        representation, i.e. no features will be extracted from the original
        images.

        If string, image features will be computed by executing:

           feature_image = eval('img.feature_type.' +
                                feature_type[level] + '()')

        for each pyramidal level. For this to work properly each string
        needs to be one of menpo's standard image feature methods
        ('igo', 'hog', ...).
        Note that, in this case, the feature computation will be
        carried out using the default options.

        Non-default feature options and new experimental features can be
        defined using functions/closures. In this case, the functions must
        receive an image as input and return a particular feature
        representation of that image. For example:

            def igo_double_from_std_normalized_intensities(image)
                image = deepcopy(image)
                image.normalize_std_inplace()
                return image.feature_type.igo(double_angles=True)

        See `menpo.image.feature.py` for details more details on
        menpo's standard image features and feature options.

        Default: None
    reference_shape: PointCloud
        The reference shape that was used to resize all training images to a
        consistent object size.
    downscale: float
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be:
            (downscale ** k) for k in range(n_levels)
    pyramid_on_features: boolean, Optional
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.
    interpolator: string
        The interpolator that was used during training.
    """
    def __init__(self, regressors, feature_type, reference_shape, downscale,
                 pyramid_on_features, interpolator):
        self._fitters = regressors
        self._feature_type = feature_type
        self._reference_shape = reference_shape
        self._downscale = downscale
        self._scaled_levels = True
        self._interpolator = interpolator
        self._pyramid_on_features = pyramid_on_features


    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the SDM family.

        : str
        """
        return 'SDM-' + self._fitters[0].algorithm

    @property
    def reference_shape(self):
        r"""
        The reference shape used during training.

        : `menpo.shape.Pointcloud`
        """
        return self._reference_shape

    @property
    def feature_type(self):
        r"""
        The feature type per pyramid level. Note that they are stored from
        lowest to highest level resolution.

        : list
        """
        return self._feature_type

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during training.

        : int
        """
        return len(self._fitters)

    @property
    def downscale(self):
        r"""
        The downscale per pyramidal level used during building the AAM.
        The scale factor is: (downscale ** k) for k in range(n_levels)

        : float
        """
        return self._downscale

    @property
    def pyramid_on_features(self):
        r"""
        Flag that controls the Gaussian pyramid of the testing image based on
        the pyramid used during building.
        If True, the feature space is computed once at the highest scale and
        the Gaussian pyramid is applied on the feature images.
        If False, the Gaussian pyramid is applied on the original images
        (intensities) and then features will be extracted at each level.

        : boolean
        """
        return self._pyramid_on_features

    @property
    def interpolator(self):
        r"""
        The employed interpolator.

        : str
        """
        return self._interpolator


class SDAAMFitter(AAMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for AAMs.

    Parameters
    -----------
    aam: :class:`menpo.fitmultilevel.aam.builder.AAM`
        The Active Appearance Model to be used.
    regressors: :class: menpo.fit.regression.RegressorTrainer
        The trained regressors.
    """
    def __init__(self, aam, regressors):
        super(SDAAMFitter, self).__init__(aam)
        self._fitters = regressors

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the SDM family.

        : str
        """
        return 'SD-AAM' + self._fitters[0].algorithm


class SDCLMFitter(CLMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for CLMs.

    Parameters
    -----------
    clm: :class:`menpo.fitmultilevel.clm.builder.CLM`
        The Constrained Local Model to be used.
    regressors: :class: menpo.fit.regression.RegressorTrainer
        The trained regressors.

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
        r"""
        Returns a string containing the algorithm used from the SDM family.

        : str
        """
        return 'SD-CLM' + self._fitters[0].algorithm
