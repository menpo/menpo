import numpy as np
from pybug.transform.base import AlignableTransform


class StatisticallyDrivenTransform(AlignableTransform):
    r"""
    A transform that couples a traditional landmark-based transform to a
    statistical model together with a global similarity transform,
    such that the parameters of the transform are fully specified by
    both the weights of statistical model and the parameters of the
    similarity transform.. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`pybug.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_constructor : func
        A function that returns a :class:`pybug.transform.base.AlignableTransform`
        object. It will be fed the source landmarks as the first
        argument and the target landmarks as the second. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    source : :class:`pybug.shape.base.PointCloud`
        The source landmarks of the transform. If no ``source`` is provided the
        mean of the model is used.
    weights : (P,) ndarray
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    """

    #TODO: Rethink this transform so it knows how to deal with complex shapes
    def __init__(self, model, transform_cls, source=None, weights=None,
                 global_transform=None, composition='both'):
        super(StatisticallyDrivenTransform, self).__init__()

        self._cached_points = None
        self.model = model
        self.global_transform = global_transform
        self.composition = composition
        if source is None:
            # set the source to the model's mean
            source = self.model.mean
        self._source = source

        if weights is None:
            # set all weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_components)
        self._weights = weights

        target = self.model.instance(self.weights)
        if self.global_transform is not None:
            target = self.global_transform.apply(target)
        self._target = target
        # by providing _source and _target we conform to the
        # AlignmentTransform interface
        # utilize the align constructor to build the transform
        self.transform = transform_cls.align(self.source, self.target)

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.transform.n_dims

    @property
    def n_parameters(self):
        r"""
        The total number of parameters.

        This is ``n_weights + n_global_parameters``.

        :type: int
        """
        return self.n_weights + self.n_global_parameters

    @property
    def n_global_parameters(self):
        r"""
        The number of parameters in the ``global_transform``

        :type: int
        """
        return self.global_transform.n_parameters

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: int
        """
        return self.model.n_components

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.global_transform.as_vector()

    @property
    def has_true_inverse(self):
        return False

    def _build_pseudoinverse(self):
        return self.from_vector(-self.as_vector())

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        r"""
        Setting the weights value automatically triggers a recalculation of
        the target.
        """
        self._weights = value
        target = self.model.instance(value)
        if self.global_transform is not None:
            target = self.global_transform.apply(target)
        self.target = target

    def jacobian(self, points):
        """
        Calculates the Jacobian of the StatisticallyDrivenTransform wrt to
        its parameters (the weights). This is done by chaining the relative
        weight of each point wrt the source landmarks, i.e. the Jacobian of
        the warp wrt the source landmarks when the target is assumed to be
        equal to the source (dW/dx), together with the Jacobian of the
        linear model (and of the global transform if present) wrt its
        weights (dX/dp).

        Parameters
        -----------
        points: (N, D) ndarray
            The points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the StatisticallyDrivenTransform evaluated at the
            previous points.
        """
        # check if re-computation of dW/dx can be avoided
        if not np.array_equal(self._cached_points, points):
            # recompute dW/dx, i.e. the relative weight of each point wrt
            # the source landmarks
            self.dW_dX = self.transform.weight_points(points)
            # cache points
            self._cached_points = points

        model_jacobian = self.model.jacobian

        # compute dX/dp
        if self.global_transform is None:
            # dX/dp is simply the Jacobian of the model
            dX_dp = model_jacobian
        else:
            # dX/dq is the Jacobian of the global transform evaluated at the
            # mean of the model.
            dX_dq = self.global_transform.jacobian(self.model.mean.points)
            # dX_dq:  n_points  x  n_global_params  x  n_dims

            # by application of the chain rule dX_db is the Jacobian of the
            # model transformed by the linear component of the global transform
            dS_db = model_jacobian
            dX_dS = self.global_transform.jacobian_points(
                self.model.mean.points)
            dX_db = np.einsum('ilj, idj -> idj', dX_dS, dS_db)
            # dS_db:  n_points  x     n_weights     x  n_dims
            # dX_dS:  n_points  x     n_dims        x  n_dims
            # dX_db:  n_points  x     n_weights     x  n_dims

            # dX/dp is simply the concatenation of the previous two terms
            dX_dp = np.hstack((dX_dq, dX_db))

        # dW_dX:    n_points   x    n_points    x  n_dims
        # dX_dp:  n_points  x     n_params      x  n_dims
        dW_dp = np.einsum('ild, lpd -> ipd', self.dW_dX, dX_dp)
        # dW_dp:    n_points   x     n_params      x  n_dims

        return dW_dp

    # TODO: document me
    def jacobian_points(self, points):
        r"""
        TO BE DOCUMENTED

        Returns
        -------
        dW_dx : (N, D, D) ndarray
            The jacobian with respect to the points
        """
        pass

    def as_vector(self):
        r"""
        Return the current parameters of this transform. This is the
        concatenated vector of the linear model's weights and the global
        transform parameters.

        Returns
        -------
        params : (``n_parameters``,) ndarray
            The vector of parameters
        """
        return np.hstack((self.global_parameters, self.weights))

    def update_from_vector(self, vector):
        r"""
        Updates the StatisticallyDrivenTransform's state from it's
        vectorized form.
        """
        self.global_transform.update_from_vector(
            vector[:self.n_global_parameters])
        # setting the weights will trigger the update to the target.
        self.weights = vector[self.n_global_parameters:]

    def _update_from_target(self, new_target):
        # TODO check this is correct
        self.global_transform.align(self.source, new_target)
        aligned_target = self.global_transform.pseudoinverse\
            .apply_nondestructive(
            new_target)
        self.weights = self.model.project(aligned_target)

    def _apply(self, x, **kwargs):
        r"""
        Apply this transform to the given object. Uses the internal transform.

        Parameters
        ----------
        x : (N, D) ndarray or a transformable object
            The object to be transformed.
        kwargs : dict
            Passed through to transforms ``apply`` method.

        Returns
        --------
        transformed : (N, D) ndarray or object
            The transformed object
        """
        return self.transform._apply(x, **kwargs)

    # TODO: Could be implemented as optimization option in LK???
    # Problems:
    #   - This method needs to be explicitly overwritten in order to match
    #     the common interface defined for AlignableTransform objects
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

    def _compose_both(self, stat_driven_transform):
        """
        Composes two statistically driven transforms together based on the
        first order approximation proposed by Papandreou and Maragos.

        Parameters
        ----------
        stat_driven_transform : :class:`StatisticallyDrivenTransform`
            The transform object to which the composition has to be
            performed with.

        Returns
        -------
        composed : :class:`StatisticallyDrivenTransform`
            The new transform representing the result of the composition.

        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        model_jacobian = self.model.jacobian

        # compute:
        # -> dW/dp when p=0
        # -> dW/dp when p!=0
        # -> dW/dx when p!=0 evaluated at the source landmarks
        if self.global_transform is None:
            # dW/dp when p=0 and when p!=0 are the same and simply given by
            # the Jacobian of the model
            dW_dp_0 = model_jacobian
            dW_dp = dW_dp_0
            # dW_dp_0:  n_points  x     n_params     x  n_dims
            # dW_dp:    n_points  x     n_params     x  n_dims
        else:
            # dW/dq when p=0 and when p!=0 are the same and given by the
            # Jacobian of the global transform evaluated at the mean of the
            # model
            dW_dq = self.global_transform.jacobian(self.model.mean.points)
            # dW_dq:  n_points  x  n_global_params  x  n_dims

            # dW/db when p=0, is the Jacobian of the model
            dW_db_0 = model_jacobian
            # dW_db_0:  n_points  x     n_weights     x  n_dims

            # dW/dp when p=0, is simply the concatenation of the previous
            # two terms
            dW_dp_0 = np.hstack((dW_dq, dW_db_0))
            # dW_dp_0:  n_points  x     n_params      x  n_dims

            # by application of the chain rule dW_db when p!=0,
            # is the Jacobian of the global transform wrt the points times
            # the Jacobian of the model: dX(S)/db = dX/dS *  dS/db
            dW_dS = self.global_transform.jacobian_points(
                self.model.mean.points)
            dW_db = np.einsum('ilj, idj -> idj', dW_dS, dW_db_0)
            # dW_dS:  n_points  x      n_dims       x  n_dims
            # dW_db:  n_points  x     n_weights     x  n_dims

            # dW/dp is simply the concatenation of dX_dq with dX_db
            dW_dp = np.hstack((dW_dq, dW_db))
            # dW_dp:    n_points  x     n_params     x  n_dims

        dW_dx = self.transform.jacobian_points(self.model.mean.points)
        #dW_dx = np.dot(dW_dx, self.global_transform.linear_component.T)
        # dW_dx:  n_points  x  n_dims  x  n_dims

        #TODO: Can we do this without splitting across the two dimensions?
        dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        dW_dp_0_mat = np.reshape(dW_dp_0, (self.model.mean.n_points * self.n_dims,
                                           self.n_parameters))
        dW_dx_dW_dp_0 = dW_dp_0_mat * dW_dx_x + dW_dp_0_mat * dW_dx_y
        dW_dx_dW_dp_0 = np.reshape(dW_dx_dW_dp_0, (self.model.mean.n_points,
                                                   self.n_parameters,
                                                   self.n_dims))
        # dW_dx:          n_points  x  n_dims    x  n_dims
        # dW_dp_0:        n_points  x  n_params  x  n_dims
        # dW_dx_dW_dp_0:  n_points  x  n_params  x  n_dims

        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)

        Jp = np.linalg.solve(H, J)
        # Jp:  n_params  x  n_params

        p = self.as_vector() + np.dot(Jp, stat_driven_transform.as_vector())

        return self.from_vector(p)
