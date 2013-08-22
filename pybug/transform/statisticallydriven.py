import numpy as np
from pybug.transform import Transform


class StatisticallyDrivenTransform(Transform):

    #TODO: Rethink this transform so it knows how to deal with complex shapes
    def __init__(self, model, transform_constructor,
                 source=None, weights=None, global_transform=None,
                 composition='model', speed_up=None):
        """
        A transform that couples a traditional landmark-based transform to a
        statistical model together with a global similarity transform,
        such that the parameters of the transform are fully specified by
        both the weights of statistical model and the parameters of the
        similarity transform.. The model is assumed to
        generate an instance which is then transformed by the similarity
        transform; the result defines the target landmarks of the transform.
        If no source is provided, the mean of the model is defined as the
        source landmarks of the transform.

        :param model: A statistical linear shape model.
        :param transform_constructor: A function that returns a Transform
            object. It will be fed the source landmarks as the first
            argument and the target landmarks as the second. The target is
            set to the points generated from the model using the
            provide weights - the source is either given or set to the
            model's mean.
        :param source: The source landmarks of the transform. If no source
            is provided the mean of the model is used.
        :param weights: The reconstruction weights that will be fed to
            the model in order to generate an instance of the target landmarks.
        """
        self.model = model
        self.transform_constructor = transform_constructor

        # source
        if source is None:
            # set the source to the model's mean
            source = self.model.mean
        self.source = source

        # weights
        if weights is None:
            # set all weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_components)
        self.weights = weights

        # global transform
        self.global_transform = global_transform

        # composition
        self.composition = composition

        # speed up
        if speed_up is not None:
            self._cached_points = speed_up[0]
            self.dW_dX = speed_up[1]
        else:
            self._cached_points = None
            self.dW_dX = None

        # generate target
        self.target = self.model.instance(self.weights)
        if self.global_transform is not None:
            self.target = self.global_transform.apply(self.target)

        # build transform
        self.transform = transform_constructor(self.source.points,
                                               self.target.points)

    @property
    def n_dim(self):
        return self.transform.n_dim

    @property
    def n_weights(self):
        return self.model.n_components

    @property
    def n_global_parameters(self):
        return self.global_transform.n_parameters

    @property
    def n_parameters(self):
        return self.n_weights + self.n_global_parameters

    @property
    def global_parameters(self):
        return self.global_transform.as_vector()

    def jacobian(self, points):
        """
        Calculates the Jacobian of the StatisticallyDrivenTransform wrt to
        its parameters (the weights). This is done by chaining the relative
        weight of each point wrt the source landmarks, i.e. the Jacobian of
        the warp wrt the source landmarks when the target is assumed to be
        equal to the source (dW/dx), together with the Jacobian of the
        linear model (and of the global transform if present) wrt its
        weights (dX/dp).
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dp: n_points x n_params x n_dims ndarray representing the
            Jacobian of the StatisticallyDrivenTransform evaluated at the
            previous points.
        """
        # check if re-computation of dW/dx can be avoided
        if not np.array_equal(self._cached_points, points):
            # recompute dW/dx, i.e. the relative weight of each point wrt
            # the source landmarks
            self.dW_dX = self.transform.weight_points(points)
            # cache points
            self._cached_points = points

        # compute dX/dp
        if self.global_transform is None:
            # dX/dp is simply the Jacobian of the model
            dX_dp = self.model.jacobian
        else:
            # dX/dq is the Jacobian of the global transform evaluated at the
            # mean of the model.
            dX_dq = self.global_transform.jacobian(self.model.mean.points)
            # dX_dq:  n_landmarks  x  n_global_params  x  n_dim

            # by application of the chain rule dX_db is the Jacobian of the
            # model transformed by the linear component of the global transform
            dS_db = self.model.jacobian
            dX_dS = self.global_transform.jacobian_points(
                self.model.mean.points)
            dX_db = np.einsum('ilj, idj -> idj', dX_dS, dS_db)
            # dS_db:  n_landmarks  x     n_weights     x  n_dim
            # dX_dS:  n_landmarks  x       n_dim       x  n_dim
            # dX_db:  n_landmarks  x     n_weights     x  n_dim

            # dX/dp is simply the concatenation of the previous two terms
            dX_dp = np.hstack((dX_dq, dX_db))

        # dW_dX:    n_points   x    n_landmarks    x  n_dim
        # dX_dp:  n_landmarks  x     n_params      x  n_dim
        dW_dp = np.einsum('ild, lpd -> ipd', self.dW_dX, dX_dp)
        # dW_dp:    n_points   x     n_params      x  n_dim

        return dW_dp

    def jacobian_points(self, points):
        pass

    def from_vector(self, flattened):
        global_transform = self.global_transform.from_vector(
            flattened[:self.n_global_parameters])
        weights = flattened[self.n_global_parameters:]

        return StatisticallyDrivenTransform(
            self.model, self.transform_constructor,
            source=self.source, weights=weights,
            global_transform=global_transform, composition=self.composition,
            speed_up=(self._cached_points, self.dW_dX))

    def as_vector(self):
        return np.hstack((self.global_parameters, self.weights))

    def _apply(self, x, **kwargs):
        return self.transform._apply(x, **kwargs)

    # TODO: Could be implemented as optimization option in LK???
    # Problems:
    #   - This method needs to be explicitly overwritten in order to match
    #     the common interface defined for Transform objects
    def compose(self, statistically_driven_transform):
        if self.composition is 'model':
            return self._compose_model(statistically_driven_transform)
        elif self.composition is 'warp':
            return self._compose_warp(statistically_driven_transform)
        elif self.composition is 'both':
            return self._compose_both(statistically_driven_transform)
        else:
            raise ValueError('Unknown composition string selected. Valid'
                             'options are: model, warp, both')

    def _compose_model(self, statistically_driven_transform):
        incremental_target = statistically_driven_transform.target
        model_variation = (self.model.instance(self.weights).points -
                           self.model.mean.points)
        composed_target = self.global_transform.apply(model_variation +
                                                      incremental_target
                                                      .points)
        from pybug.shape import PointCloud
        return self.estimate(PointCloud(composed_target))

    # TODO: The call to transform.apply will not work properly for PWA
    #   - Define a new function in TPS & PWA called .apply_to_target
    #   - For TPS this function should ne the same as the normal .apply()
    #     method
    #   - For PWA it should implement Bakers algorithmic approach to
    #     composition
    def _compose_warp(self, statistically_driven_transform):
        incremental_target = statistically_driven_transform.target
        composed_target = self.transform.apply(incremental_target)

        return self.estimate(composed_target)

    def _compose_both(self, statistically_driven_transform):
        """
        Composes two statistically driven transforms together based on the
        first order approximation proposed in:

        - G. Papandreou and P. Maragos, "Adaptive and Constrained Algorithms
          for Inverse Compositional Active Appearance Model Fitting", CVPR08

        :param statistically_driven_transform: the StatisticallyDrivenTransform
            object to which the composition has to be performed with.
        :return the resulting StatisticallyDrivenTransform
        """
        # compute:
        # -> dW/dp when p=0
        # -> dW/dp when p!=0
        # -> dW/dx when p!=0 evaluated at the source landmarks
        if self.global_transform is None:
            # dW/dp when p=0 and when p!=0 are the same and simply given by
            # the Jacobian of the model
            dW_dp_0 = self.model.jacobian
            dW_dp = dW_dp_0
            # dW_dp_0:  n_landmarks  x     n_params     x  n_dim
            # dW_dp:    n_landmarks  x     n_params     x  n_dim
        else:
            # dW/dq when p=0 and when p!=0 are the same and given by the
            # Jacobian of the global transform evaluated at the mean of the
            # model
            dW_dq = self.global_transform.jacobian(self.model.mean.points)
            # dW_dq:  n_landmarks  x  n_global_params  x  n_dim

            # dW/db when p=0, is the Jacobian of the model
            dW_db_0 = self.model.jacobian
            # dW_db_0:  n_landmarks  x     n_weights     x  n_dim

            # dW/dp when p=0, is simply the concatenation of the previous
            # two terms
            dW_dp_0 = np.hstack((dW_dq, dW_db_0))
            # dW_dp_0:  n_landmarks  x     n_params      x  n_dim

            # by application of the chain rule dW_db when p!=0,
            # is the Jacobian of the global transform wrt the points times
            # the Jacobian of the model: dX(S)/db = dX/dS *  dS/db
            dW_dS = self.global_transform.jacobian_points(
                self.model.mean.points)
            dW_db = np.einsum('ilj, idj -> idj', dW_dS, dW_db_0)
            # dW_dS:  n_landmarks  x      n_dims       x  n_dim
            # dW_db:  n_landmarks  x     n_weights     x  n_dim

            # dW/dp is simply the concatenation of dX_dq with dX_db
            dW_dp = np.hstack((dW_dq, dW_db))
            # dW_dp:    n_landmarks  x     n_weights     x  n_dim

        dW_dx = self.transform.jacobian_points(self.source)
        #dW_dx = np.dot(dW_dx, self.global_transform.linear_component.T)
        # dW_dx:  n_landmarks  x  n_dim  x  n_dim

        dW_dx_dW_dp_0 = np.einsum('ijl, idl -> idj', dW_dx, dW_dp_0)
        # dW_dx:          n_landmarks  x  n_dim     x  n_dim
        # dW_dp_0:        n_landmarks  x  n_params  x  n_dim
        # dW_dx_dW_dp_0:  n_landmarks  x  n_params  x  n_dim

        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)

        Jp = np.linalg.solve(H, J)
        # Jp:  n_params  x  n_params

        p = (self.as_vector() +
             np.sum(Jp * statistically_driven_transform.as_vector(),
                    axis=0))

        return self.from_vector(p)

    def estimate(self, target):
        global_transform = self.global_transform.estimate(
            self.model.mean.points, target.points)
        global_parameters = global_transform.as_vector()
        aligned_target = global_transform.inverse.apply(target)
        weights = self.model.project(aligned_target)
        parameters = np.hstack((global_parameters, weights))
        return self.from_vector(parameters)

    @property
    def inverse(self):
        return self.from_vector(-self.as_vector())