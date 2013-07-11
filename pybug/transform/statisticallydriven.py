import numpy as np
from pybug.transform import Transform


class StatisticallyDrivenTransform(Transform):

    def __init__(self, model, transform_constructor, weights=None):
        """
        A transform that couples a traditional landmark-based transform to a
        statistical model, such that the parameters of the model are fully
        specified by the statistical model that drives the transform. The
        model is assumed to generate instance which dictate of the source
        landmarks of the transform. The mean of the model is always the
        target landmarks of the model.

        :param model: A statistical shape model.
        :param transform_constructor: A function that returns a Transform
            object. It will be fed the source landmarks as the first
            argument and the target landmarks as the second. The target
            landmarks are always the model's mean - the source is set to the
            points generated from the model using the weights provided.
        :param weights: The reconstruction weights that will be fed to the
            model in order to generate an instance of the target landmarks.
        """
        self.model = model
        if weights is None:
            # set all the weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_components)
        self.weights = weights
        # TODO we need an example of a transform_constructor.
        self.transform_constructor = transform_constructor
        self.transform = transform_constructor(
            self.model.instance(weights), self.model.mean)

    def jacobian(self, points):
        dW_dX = self.transform.jacobian_shape(points)
        # TODO the model needs to be able to generate it's jacobian.
        dX_dP = self.model.jacobian(points)
        # TODO these need to be chained together property
        return dW_dX, dX_dP

    def from_vector(self, flattened):
        return StatisticallyDrivenTransform(self.model,
                                            self.transform_constructor,
                                            weights=flattened)

    def as_vector(self):
        return self.weights