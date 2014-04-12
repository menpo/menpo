import abc

from menpo.fit.base import Fitter
from menpo.fit.fittingresult import (NonParametricFittingResult,
                                     SemiParametricFittingResult,
                                     ParametricFittingResult)


#TODO: document me
class Regressor(Fitter):

    def __init__(self, regressor, features):
        self.regressor = regressor
        self.features = features

    def _set_up(self):
        pass

    def _fit(self, fitting, max_iters=1):
        image = fitting.image
        initial_shape = fitting.initial_shape
        n_iters = 0

        while n_iters < max_iters:
            features = self.features(image, initial_shape)
            delta_p = self.regressor(features)

            fitted_shape, parameters = self.update(delta_p, initial_shape)
            fitting.parameters.append(parameters)
            n_iters += 1

        fitting.fitted = True
        return fitting

    @abc.abstractmethod
    def update(self, delta_p, initial_shape):
        pass


#TODO: document me
class NonParametricRegressor(Regressor):

    def __init__(self, regressor, features):
        super(NonParametricRegressor, self).__init__(
            regressor, features)

    @property
    def algorithm(self):
        return "Non-Parametric"

    def _create_fitting(self, image, shape, gt_shape=None):
        return NonParametricFittingResult(image, self, shape=[shape],
                                          gt_shape=gt_shape)

    def update(self, delta_shape, initial_shape):
        fitted_shape = initial_shape.from_vector(
            initial_shape.as_vector() + delta_shape)
        return fitted_shape, fitted_shape


#TODO: Document me
class SemiParametricRegressor(Regressor):

    def __init__(self, regressor, features, transform, update='composition'):
        super(SemiParametricRegressor, self).__init__(
            regressor, features)
        self.transform = transform
        self._update = self._select_update(update)

    @property
    def algorithm(self):
        return "SemiParametric"

    def _create_fitting(self, image, shape, gt_shape=None):
        self.transform.set_target(shape)
        return SemiParametricFittingResult(
            image, self, parameters=[self.transform.as_vector()],
            gt_shape=gt_shape)

    def _select_update(self, update):
        if update is 'additive':
            return self._additive
        elif update is 'compositional':
            return self._compositional
        else:
            raise ValueError('Unknown update string selected. Valid'
                             'options are: additive, compositional')

    def _additive(self, delta_p):
        parameters = self.transform.as_vector() + delta_p
        self.transform.from_vector_inplace(parameters)

    def _compositional(self, delta_p):
        self.transform.compose_after_from_vector_inplace(delta_p)

    def update(self, delta_p, initial_shape):
        self._update(delta_p)
        return self.transform.target, self.transform.as_vector()


#TODO: Document me
class ParametricRegressor(SemiParametricRegressor):

    def __init__(self, regressor, features, appearance_model, transform,
                 update='composition'):
        super(ParametricRegressor, self).__init__(
            regressor, features, transform, update=update)
        self.appearance_model = appearance_model
        self.template = appearance_model.mean

    @property
    def algorithm(self):
        return "Parametric"

    def _create_fitting(self, image, shape, gt_shape=None):
        self.transform.set_target(shape)
        return ParametricFittingResult(
            image, self, parameters=[self.transform.as_vector()],
            gt_shape=gt_shape)
