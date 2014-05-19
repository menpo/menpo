import abc

from menpo.fit.base import Fitter
from menpo.fit.fittingresult import (NonParametricFittingResult,
                                     SemiParametricFittingResult,
                                     ParametricFittingResult)


class Regressor(Fitter):
    r"""
    An abstract base class for fitting regressors.

    Parameters
    ----------
    regressor:
        The regressor to be used.
    features:
        The features used to regress.
    """
    def __init__(self, regressor, features):
        self.regressor = regressor
        self.features = features

    def _set_up(self):
        r"""
        Abstract method that sets up the fitter object.
        """
        pass

    def _fit(self, fitting_result, max_iters=1):
        r"""
        Abstract method to fit an image.

        Parameters
        ----------
        fitting_result: `menpo.fit.fittingresult`
            The fitting result object.
        max_iters: int
            The maximum number of iterations.
        """
        image = fitting_result.image
        initial_shape = fitting_result.initial_shape
        n_iters = 0

        while n_iters < max_iters:
            features = self.features(image, initial_shape)
            delta_p = self.regressor(features)

            fitted_shape, parameters = self.update(delta_p, initial_shape)
            fitting_result.parameters.append(parameters)
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result

    @abc.abstractmethod
    def update(self, delta_p, initial_shape):
        r"""
        Abstract method to update the parameters.
        """
        pass


class NonParametricRegressor(Regressor):
    r"""
    Fitter of Non-Parametric Regressor.

    Parameters
    ----------
    regressor:
        The regressor to be used.
    features:
        The features used to regress.
    """
    def __init__(self, regressor, features):
        super(NonParametricRegressor, self).__init__(
            regressor, features)

    @property
    def algorithm(self):
        r"""
        Returns the regression type.
        """
        return "Non-Parametric"

    def _create_fitting_result(self, image, shape, gt_shape=None):
        r"""
        Creates the fitting result object.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image..
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        gt_shape: :class:`menpo.shape.PointCloud`
            The ground truth shape.
        """
        return NonParametricFittingResult(image, self, shape=[shape],
                                          gt_shape=gt_shape)

    def update(self, delta_shape, initial_shape):
        fitted_shape = initial_shape.from_vector(
            initial_shape.as_vector() + delta_shape)
        return fitted_shape, fitted_shape


class SemiParametricRegressor(Regressor):
    r"""
    Fitter of Semi-Parametric Regressor.

    Parameters
    ----------
    regressor:
        The regressor to be used.
    features:
        The features used to regress.
    """
    def __init__(self, regressor, features, transform, update='composition'):
        super(SemiParametricRegressor, self).__init__(
            regressor, features)
        self.transform = transform
        self._update = self._select_update(update)

    @property
    def algorithm(self):
        r"""
        Returns the regression type.
        """
        return "Semi-Parametric"

    def _create_fitting_result(self, image, shape, gt_shape=None):
        r"""
        Creates the fitting result object.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image..
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        gt_shape: :class:`menpo.shape.PointCloud`, Optional
            The ground truth shape.

            Default: None
        """
        self.transform.set_target(shape)
        return SemiParametricFittingResult(
            image, self, parameters=[self.transform.as_vector()],
            gt_shape=gt_shape)

    def fit(self, image, initial_parameters, gt_shape=None, **kwargs):
        self.transform.from_vector_inplace(initial_parameters)
        return Fitter.fit(self, image, initial_parameters, gt_shape=gt_shape,
                          **kwargs)

    def _select_update(self, update):
        r"""
        Select the way to update the parameters.

        Parameter
        ---------
        update: 'compositional' or 'additive'

        Returns
        -------
        : function/closure
        """
        if update is 'additive':
            return self._additive
        elif update is 'compositional':
            return self._compositional
        else:
            raise ValueError('Unknown update string selected. Valid'
                             'options are: additive, compositional')

    def _additive(self, delta_p):
        r"""
        Updates the parameters in the additive way.

        Parameter
        ---------
        delta_p:
            The parameters increment
        """
        parameters = self.transform.as_vector() + delta_p
        self.transform.from_vector_inplace(parameters)

    def _compositional(self, delta_p):
        r"""
        Updates the parameters in the compositional way.

        Parameter
        ---------
        delta_p:
            The parameters increment
        """
        self.transform.compose_after_from_vector_inplace(delta_p)

    def update(self, delta_p, initial_shape):
        r"""
        Updates the parameters.

        Parameters
        ----------
        delta_p:
            The parameters increment.
        initial_shape: PointCloud
            The initial shape.
        """
        self._update(delta_p)
        return self.transform.target, self.transform.as_vector()


class ParametricRegressor(SemiParametricRegressor):
    r"""
    Fitter of Parametric Regressor.

    Parameters
    ----------
    regressor:
        The regressor to be used.
    features:
        The features used to regress.
    """
    def __init__(self, regressor, features, appearance_model, transform,
                 update='composition'):
        super(ParametricRegressor, self).__init__(
            regressor, features, transform, update=update)
        self.appearance_model = appearance_model
        self.template = appearance_model.mean

    @property
    def algorithm(self):
        r"""
        Returns the regression type.
        """
        return "Parametric"

    def _create_fitting_result(self, image, shape, gt_shape=None):
        r"""
        Creates the fitting result object.

        Parameters
        ----------
        image: :class:`menpo.image.MaskedImage`
            The current image..
        shape: :class:`menpo.shape.PointCloud`
            The current shape.
        gt_shape: :class:`menpo.shape.PointCloud`, Optional
            The ground truth shape.

            Default: None
        """
        self.transform.set_target(shape)
        return ParametricFittingResult(
            image, self, parameters=[self.transform.as_vector()],
            gt_shape=gt_shape)
