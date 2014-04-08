import abc


#TODO: document me
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
        """
        fitting = self._create_fitting(image, initial_parameters,
                                       gt_shape=gt_shape)
        return self._fit(fitting, **kwargs)

    @abc.abstractmethod
    def _create_fitting(self, **kwargs):
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
        """
        return shape
