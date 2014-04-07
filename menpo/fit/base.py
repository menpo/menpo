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
        """
        pass

    @abc.abstractmethod
    def _set_up(self, **kwargs):
        r"""
        Sets up the Fitter object. Usually performs pre-computations related
        to specific alignment algorithms used by the Fitter object. Highly
        dependent on the type of fitter object.
        """
        pass

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        r"""
        """
        fitting = self._create_fitting(image, initial_parameters,
                                       gt_shape=gt_shape)
        return self._fit(fitting, **kwargs)

    @abc.abstractmethod
    def _create_fitting(self, **kwargs):
        r"""
        """
        pass

    @abc.abstractmethod
    def _fit(self, **kwargs):
        r"""
        Abstract method to be overwritten by subclasses that implements the
        alignment algorithm.
        """
        pass

    def get_parameters(self, shape):
        r"""
        """
        return shape