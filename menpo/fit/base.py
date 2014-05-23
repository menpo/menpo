import abc


class Fitter(object):
    r"""
    Interface that all Fitter objects must implement.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def algorithm(self):
        r"""
        Returns the name of the fitter object.
        """
        pass

    @abc.abstractmethod
    def _set_up(self, **kwargs):
        r"""
        Abstract method that sets up the fitter object.
        """
        pass

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        r"""
        Fits the fitter object to an image.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The image to be fitted.
        initial_parameters: list
            The initial parameters of the model.
        gt_shape: :class:`menpo.shape.PointCloud`, optional
            The original ground truth shape associated to the image.

            Default: None

        Returns
        -------
        fitting_result: `menpo.fit.fittingresult`
            The fitting result.
        """
        fitting_result = self._create_fitting_result(
            image, initial_parameters, gt_shape=gt_shape)
        return self._fit(fitting_result, **kwargs)

    @abc.abstractmethod
    def _create_fitting_result(self, **kwargs):
        r"""
        Abstract method that defines the fitting result object associated to
        the fitter object.
        """
        pass

    @abc.abstractmethod
    def _fit(self, **kwargs):
        r"""
        Abstract method implements a particular alignment algorithm.
        """
        pass

    def get_parameters(self, shape):
        r"""
        Abstract method that gets the parameters.
        """
        pass
