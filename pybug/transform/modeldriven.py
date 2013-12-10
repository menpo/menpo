from copy import deepcopy
import numpy as np
from pybug.model import Similarity2dInstanceModel
from pybug.transform.base import AlignableTransform, Composable


class ModelDrivenTransform(AlignableTransform, Composable):
    r"""
    A transform that couples a traditional landmark-based transform to a
    statistical model such that source points of the alignment transform
    are the points of the model. The parameters of the transform are just
    the weights of statistical model.

    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`pybug.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`pybug.transform.AlignableTransform`
        A class of :class:`pybug.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    source : :class:`pybug.shape.base.PointCloud`
        The source landmarks of the transform. If no ``source`` is provided the
        mean of the model is used.
    weights : (P,) ndarray
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    composition: 'both', 'warp' or 'model', optional
        The composition approximation employed by this
        ModelDrivenTransform.

        Default: 'both'
    """
    #TODO: Rethink this transform so it knows how to deal with complex shapes
    def __init__(self, model, transform_cls, source=None, weights=None,
                 composition='both'):
        super(ModelDrivenTransform, self).__init__()

        self._cached_points = None
        self.model = model
        self.composition = composition
        if source is None:
            # set the source to the model's mean
            source = self.model.mean
        self._source = source

        if weights is None:
            # set all weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_active_components)
        self._weights = weights

        self._target = self._target_for_weights(self._weights)
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

        Simply ``n_weights``.

        :type: int
        """
        return self.n_weights

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: int
        """
        return self.model.n_active_components

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
        the target, and an update of the transform
        """
        self.target = self._target_for_weights(value)

    def jacobian(self, points):
        """
        Calculates the Jacobian of the ModelDrivenTransform wrt to
        its parameters (the weights). This is done by chaining the relative
        weight of each point wrt the source landmarks, i.e. the Jacobian of
        the warp wrt the source landmarks when the target is assumed to be
        equal to the source (dW/dx), together with the Jacobian of the
        linear model  wrt its weights (dX/dp).

        Parameters
        -----------
        points: (N, D) ndarray
            The points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the ModelDrivenTransform evaluated at the
            previous points.
        """
        # check if re-computation of dW/dx can be avoided
        if not np.array_equal(self._cached_points, points):
            # recompute dW/dx, i.e. the relative weight of each point wrt
            # the source landmarks
            self.dW_dX = self.transform.weight_points(points)
            # cache points
            self._cached_points = points

        # dX/dp is simply the Jacobian of the model
        dX_dp = self.model.jacobian

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
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (``n_parameters``,) ndarray
            The vector of parameters
        """
        return self.weights

    def from_vector_inplace(self, vector):
        r"""
        Updates the ModelDrivenTransform's state from it's
        vectorized form.
        """
        self.weights = vector

    def _target_setter(self, new_target):
        r"""
        On a new target being set, we need to:

        1. Find the optimum weights that align the model to this target,
        and set them as self.weights.

        2. Update the transform to point to the closest target that the
        model can provide to the requested target

        3. Set our target to the closest target that the model can provide
        to the requested target.

        Parameters
        ----------

        new_target: :class:`PointCloud`
            The new_target that we want to set.
        """
        # 1. Find the optimum weights and set them
        self._weights = self._weights_for_target(new_target)
        # 2. Find the closest target the model can reproduce and trigger an
        # update of our transform
        self.transform.target = self._target_for_weights(self._weights)
        # 3. As always, update our self._target
        self._target = self.transform.target

    def _apply(self, x, **kwargs):
        r"""
        Apply this transform to the given object. Uses the internal transform.

        Parameters
        ----------
        x : (N, D) ndarray or a transformable object
            The object to be transformed.
        kwargs : dict
            Passed through to transforms ``apply_inplace`` method.

        Returns
        --------
        transformed : (N, D) ndarray or object
            The transformed object
        """
        return self.transform._apply(x, **kwargs)

    def compose_before_inplace(self, transform):
        r"""
        a_orig = deepcopy(a)
        a.compose_before_inplace(b)
        a.apply(p) == b.apply(a_orig.apply(p))

        a is permanently altered to be the result of the composition. b is
        left unchanged.

        Parameters
        ----------
        transform : :class:`ModelDrivenTransform`
            Transform to be applied **after** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        # naive approach - update self to be equal to transform and
        # compose_before_from_vector_inplace
        self_vector = self.as_vector().copy()
        self.update_from_vector(transform.as_vector())
        return self.compose_after_from_vector_inplace(self_vector)

    def compose_after(self, transform):
        r"""
        c = a.compose_after(b)
        c.apply(p) == a.apply(b.apply(p))

        a and b are left unchanged.

        This corresponds to the usual mathematical formalism for the compose
        operator, `o`.

        Parameters
        ----------
        transform : :class:`ModelDrivenTransform`
            Transform to be applied **before** self

        Returns
        --------
        transform : :class:`ModelDrivenTransform`
            The resulting ModelDrivenTransform.
        """
        self_copy = deepcopy(self)
        self_copy.compose_after_inplace(transform)
        return self_copy

    def compose_after_inplace(self, md_transform):
        r"""
        a_orig = deepcopy(a)
        a.compose_after_inplace(b)
        a.apply(p) == a_orig.apply(b.apply(p))

        a is permanently altered to be the result of the composition. b is
        left unchanged.

        Parameters
        ----------
        transform : :class:`ModelDrivenTransform`
            Transform to be applied **before** self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        if self.composition is 'model':
            # TODO this seems to be the same, revisit
            self.target = self._compose_after_model(md_transform.target)
        elif self.composition is 'warp':
            self.target = self._compose_after_warp(md_transform.target)
        elif self.composition is 'both':
            new_params = self._compose_after_both(md_transform.as_vector())
            self.from_vector_inplace(new_params)
        else:
            raise ValueError('Unknown composition string selected. Valid'
                             'options are: model, warp, both')
        return self

    def compose_after_from_vector_inplace(self, vector):
        r"""
        a_orig = deepcopy(a)
        a.compose_after_from_vector_inplace(b_vec)
        b = self.from_vector(b_vec)
        a.apply(p) == a_orig.apply(b.apply(p))

        a is permanently altered to be the result of the composition. b_vec
        is left unchanged.

        compose_after this :class:`ModelDrivenTransform` with another inplace.
        Rather than requiring a new ModelDrivenTransform to compose_after
        with, this method only requires the parameters of the new transform.

        Parameters
        ----------
        vector : (N,) ndarray
            Vectorized :class:`ModelDrivenTransform` to be applied **before**
            self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        if self.composition is 'model':
            new_mdtransform = self.from_vector(vector)
            self.target = self._compose_after_model(new_mdtransform.target)
        elif self.composition is 'warp':
            new_mdtransform = self.from_vector(vector)
            self.target = self._compose_after_warp(new_mdtransform.target)
        elif self.composition is 'both':
            self.from_vector_inplace(self._compose_after_both(vector))
        else:
            raise ValueError('Unknown composition string selected. Valid'
                             'options are: model, warp, both')

    def _compose_after_model(self, other_target):
        r"""
        Composes two statistically driven transforms together.

        Parameters
        ----------
        other_target : :class:`PointCloud`
            the target of the ModelDrivenTransform we are
            composing with.

        Returns
        -------

        target: :class:`PointCloud`
            The new target of the composed result
        """
        model_variation = self.target.points - self.model.mean.points
        composed_target = model_variation + other_target.points
        from pybug.shape import PointCloud
        return PointCloud(composed_target)

    # TODO: The call to transform.apply_inplace will not work properly for PWA
    #   - Define a new function in TPS & PWA called .apply_to_target
    #   - For TPS this function should ne the same as the normal .apply_inplace()
    #     method
    #   - For PWA it should implement Bakers algorithmic approach to
    #     composition
    def _compose_after_warp(self, other_target):
        r"""
        Composes two statistically driven transforms together. This approach
        composes the

        Parameters
        ----------
        other_target : :class:`PointCloud`
            the target of the ModelDrivenTransform we are
            composing with.

        Returns
        -------

        target: :class:`PointCloud`
            The new target of the composed result
        """
        return self.transform.apply(other_target)

    def _compose_after_both(self, mdt_vector):
        r"""
        Composes two statistically driven transforms together based on the
        first order approximation proposed by Papandreou and Maragos.

        The resulting vector of parameters is equivalent to

            self.compose_after_from_vector(mdt_vector)

        Parameters
        ----------
        mdt_vector : (P,) ndarray
            the parameters of the ModelDrivenTransform we are
            composing with, as provided by .as_vector().

        Returns
        -------
        vector: (P,) ndarray
            The new parameters of the composed result

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

        # dW/dp when p=0 and when p!=0 are the same and simply given by
        # the Jacobian of the model
        dW_dp_0 = model_jacobian
        dW_dp = dW_dp_0
        # dW_dp_0:  n_points  x     n_params     x  n_dims
        # dW_dp:    n_points  x     n_params     x  n_dims

        dW_dx = self.transform.jacobian_points(self.model.mean.points)
        # dW_dx:  n_points  x  n_dims  x  n_dims

        #TODO: Can we do this without splitting across the two dimensions?
        dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        dW_dp_0_mat = np.reshape(dW_dp_0, (self.model.mean.n_points *
                                           self.n_dims, self.n_parameters))
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

        return self.as_vector() + np.dot(Jp, mdt_vector)

    def _target_for_weights(self, weights):
        r"""
        Return the appropriate target for the model weights provided.
        Subclasses can override this.

        Parameters
        ----------

        weights: (P,) ndarray
            weights of the statistical model that should be used to generate a
            new instance

        Returns
        -------

        new_target: :class:`pybug.shape.PointCloud`
            A new target for the weights provided
        """
        return self.model.instance(weights)

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided.
        Subclasses can override this.

        Parameters
        ----------

        target: :class:`pybug.shape.PointCloud`
            The target that the statistical model will try to reproduce

        Returns
        -------

        weights: (P,) ndarray
            Weights of the statistical model that generate the closest
            PointCloud to the requested target
        """
        return self.model.project(target)

    def pseudoinverse_vector(self, vector):
        r"""
        The vectorized pseudoinverse of a provided vector instance.

        Syntactic sugar for

        self.from_vector(vector).pseudoinverse.as_vector()

        On ModelDrivenTransform this is especially fast - we just negate the
        vector provided.

        Parameters
        ----------
        vector :  (P,) ndarray
            A vectorized version of self

        Returns
        -------
        pseudoinverse_vector : (N,) ndarray
            The pseudoinverse of the vector provided
        """
        # just have to negate the parameters!
        return -vector


class GlobalMDTransform(ModelDrivenTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the parameters of the transform are fully specified by
    both the weights of statistical model and the parameters of the
    similarity transform. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`pybug.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`pybug.transform.AlignableTransform`
        A class of :class:`pybug.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`pybug.transform.AlignableTransform`
        A class of :class:`pybug.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`pybug.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no ``source`` is provided the
        mean of the model is used.
    weights : (P,) ndarray, optional
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    composition: 'both', 'warp' or 'model', optional
        The composition approximation employed by this
        ModelDrivenTransform.

        Default: `both`
    """
    def __init__(self, model, transform_cls, global_transform, source=None,
                 weights=None, composition='both'):
        # need to set the global transform right away - self
        # ._target_for_weights() needs it in superclass __init__
        self.global_transform = global_transform
        super(GlobalMDTransform, self).__init__(
            model, transform_cls, source=source, weights=weights,
            composition=composition)
        # after construction, we want our global_transform() to be an align
        # transform. This is a little hacky, but is ok as long as the
        # superclasses __init__ doesn't use _weights_for_target.
        self.global_transform = global_transform.align(self.model.mean,
                                                       self.target)

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
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.global_transform.as_vector()

    def jacobian(self, points):
        """
        Calculates the Jacobian of the ModelDrivenTransform wrt to
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
            The Jacobian of the ModelDrivenTransform evaluated at the
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

        # dX/dq is the Jacobian of the global transform evaluated at the
        # mean of the model.
        dX_dq = self._global_transform_jacobian(self.model.mean.points)
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

    def _global_transform_jacobian(self, points):
        return self.global_transform.jacobian(points)

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

    def from_vector_inplace(self, vector):
        # the only extra step we have to take in
        global_params = vector[:self.n_global_parameters]
        model_params = vector[self.n_global_parameters:]
        self._update_global_weights(global_params)
        self.weights = model_params

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform.from_vector_inplace(global_weights)

    def _compose_after_model(self, other_target):
        r"""
        Composes two statistically driven transforms together.

        Parameters
        ----------
        other_target : :class:`PointCloud`
            the target of the ModelDrivenTransform we are
            composing with.

        Returns
        -------

        target: :class:`PointCloud`
            The new target of the composed result
        """
        model_variation = (
            self.global_transform.pseudoinverse.apply(self.target.points) -
            self.model.mean.points)
        composed_target = self.global_transform.apply(
            model_variation + other_target.points)
        from pybug.shape import PointCloud
        return PointCloud(composed_target)

    def _compose_after_both(self, mdt_vector):
        r"""
        Composes two statistically driven transforms together based on the
        first order approximation proposed by Papandreou and Maragos.

        Parameters
        ----------
        new_sdt_parameters : (P,) ndarray
            the parameters of the ModelDrivenTransform we are
            composing with, as provided by .as_vector().

        Returns
        -------

        parameters: (P,) ndarray
            The new parameters of the composed result

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

        # dW/dq when p=0 and when p!=0 are the same and given by the
        # Jacobian of the global transform evaluated at the mean of the
        # model
        dW_dq = self._global_transform_jacobian(self.model.mean.points)
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
        dW_dS = self.global_transform.jacobian_points(self.model.mean.points)
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

        return self.as_vector() + np.dot(Jp, mdt_vector)

    def _target_for_weights(self, weights):
        r"""
        Return the appropriate target for the model weights provided,
        accounting for the effect of the global transform

        Parameters
        ----------

        weights: (P,) ndarray
            weights of the statistical model that should be used to generate a
            new instance

        Returns
        -------

        new_target: :class:`pybug.shape.PointCloud`
            A new target for the weights provided
        """
        return self.global_transform.apply(self.model.instance(weights))

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided, accounting
        for the effect of the global transform. Note that this method
        updates the global transform to be in the correct state.

        Parameters
        ----------

        target: :class:`pybug.shape.PointCloud`
            The target that the statistical model will try to reproduce

        Returns
        -------

        weights: (P,) ndarray
            Weights of the statistical model that generate the closest
            PointCloud to the requested target
        """

        self._update_global_transform(target)
        projected_target = self.global_transform.pseudoinverse.apply(target)
        # now we have the target in model space, project it to recover the
        # weights
        new_weights = self.model.project(projected_target)
        # TODO investigate the impact of this, could be problematic
        # the model can't perfectly reproduce the target we asked for -
        # reset the global_transform.target to what it CAN produce
        #refined_target = self._target_for_weights(new_weights)
        #self.global_transform.target = refined_target
        return new_weights

    def _update_global_transform(self, target):
        self.global_transform.target = target


class OrthoMDTransform(GlobalMDTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the parameters of the transform are fully specified by
    both the weights of statistical model and the parameters of the
    similarity transform. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    This transform (in contrast to the :class:`GlobalMDTransform`)
    additionally orthonormalizes both the global and the model basis against
    each other, ensuring that orthogonality and normalization is enforced
    across the unified bases.

    Parameters
    ----------
    model : :class:`pybug.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`pybug.transform.AlignableTransform`
        A class of :class:`pybug.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`pybug.transform.AlignableTransform`
        A class of :class:`pybug.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`pybug.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no ``source`` is provided the
        mean of the model is used.
    weights : (P,) ndarray, optional
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    composition: 'both', 'warp' or 'model', optional
        The composition approximation employed by this
        ModelDrivenTransform.

        Default: `both`
    """
    def __init__(self, model, transform_cls, global_transform, source=None,
                 weights=None, composition='both'):
        # 1. Construct similarity model from the mean of the model
        self.similarity_model = Similarity2dInstanceModel(model.mean)
        # 2. Orthonormalize model and similarity model
        model = deepcopy(model)
        model.orthonormalize_against_inplace(self.similarity_model)
        self.similarity_weights = self.similarity_model.project(
            global_transform.apply(model.mean))

        super(OrthoMDTransform, self).__init__(
            model, transform_cls, global_transform, source=source,
            weights=weights, composition=composition)

    def _update_global_transform(self, target):
        self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.target = new_target

    def _global_transform_jacobian(self, points):
        return self.similarity_model.jacobian

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.similarity_weights
