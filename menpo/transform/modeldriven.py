import numpy as np

from menpo.base import Targetable, Vectorizable
from menpo.model.pdm import PDM, GlobalPDM, OrthoPDM

from .base import Transform, VComposable, VInvertible


class ModelDrivenTransform(Transform, Targetable, Vectorizable,
                           VComposable, VInvertible):
    r"""
    A transform that couples a traditional landmark-based transform to a
    statistical model such that source points of the alignment transform
    are the points of the model. The weights of the transform are just
    the weights of statistical model.

    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    source : :class:`menpo.shape.base.PointCloud`
        The source landmarks of the transform. If None, the mean of the model
         is used.

        Default: None

    """
    def __init__(self, model, transform_cls, source=None):
        self.pdm = PDM(model)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.pdm.n_dims

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

    @property
    def target(self):
        return self.pdm.target

    def _target_setter(self, new_target):
        r"""
        On a new target being set, we need to:

        Parameters
        ----------

        new_target: :class:`PointCloud`
            The new_target that we want to set.
        """
        self.pdm.set_target(new_target)

    def _new_target_from_state(self):
        # We delegate to PDM to handle all our Targetable duties. As a
        # result, *we* never need to call _sync_target_for_state, so we have
        # no need for an implementation of this method. Of course the
        # interface demands it, so the stub is here. Contrast with
        # _target_setter, which is required, because we will have to handle
        # external calls to set_target().
        pass

    def _sync_state_from_target(self):
        # Let the pdm update its state
        self.pdm._sync_state_from_target()
        # and update our transform to the new state
        self.transform.set_target(self.target)

    @property
    def n_parameters(self):
        r"""
        The total number of parameters.

        Simply ``n_weights``.

        :type: int
        """
        return self.pdm.n_parameters

    def as_vector(self):
        r"""
        Return the current weights of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (``n_parameters``,) ndarray
            The vector of weights
        """
        return self.pdm.as_vector()

    def from_vector_inplace(self, vector):
        r"""
        Updates the ModelDrivenTransform's state from it's
        vectorized form.
        """
        self.pdm.from_vector_inplace(vector)
        # By here the pdm has updated our target state, we just need to
        # update the transform
        self.transform.set_target(self.target)

    def compose_after_from_vector_inplace(self, delta):
        r"""
        Composes two ModelDrivenTransforms together based on the
        first order approximation proposed by Papandreou and Maragos in [1].

        Parameters
        ----------
        delta : (N,) ndarray
            Vectorized :class:`ModelDrivenTransform` to be applied **before**
            self

        Returns
        --------
        transform : self
            self, updated to the result of the composition


        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        model_jacobian = self.pdm.model.jacobian
        points = self.pdm.model.mean.points
        n_points = self.pdm.model.mean.n_points

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

        dW_dx = self.transform.jacobian_points(points)
        # dW_dx:  n_points  x  n_dims  x  n_dims

        #TODO: Can we do this without splitting across the two dimensions?
        dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        dW_dp_0_mat = np.reshape(dW_dp_0, (n_points * self.n_dims,
                                           self.n_parameters))
        dW_dx_dW_dp_0 = dW_dp_0_mat * dW_dx_x + dW_dp_0_mat * dW_dx_y
        dW_dx_dW_dp_0 = np.reshape(dW_dx_dW_dp_0,
                                   (n_points, self.n_parameters, self.n_dims))
        # dW_dx:          n_points  x  n_dims    x  n_dims
        # dW_dp_0:        n_points  x  n_params  x  n_dims
        # dW_dx_dW_dp_0:  n_points  x  n_params  x  n_dims

        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)

        Jp = np.linalg.solve(H, J)
        # Jp:  n_params  x  n_params

        self.from_vector_inplace(self.as_vector() + np.dot(Jp, delta))
        return self

    @property
    def has_true_inverse(self):
        return False

    def _build_pseudoinverse(self):
        return self.from_vector(-self.as_vector())

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
        return -vector

    def jacobian(self, points):
        """
        Calculates the Jacobian of the ModelDrivenTransform wrt to
        its weights (the weights). This is done by chaining the relative
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
        dX_dp = self.pdm.model.jacobian

        # dW_dX:    n_points   x    n_points    x  n_dims
        # dX_dp:  n_points  x     n_params      x  n_dims
        dW_dp = np.einsum('ild, lpd -> ipd', self.dW_dX, dX_dp)
        # dW_dp:    n_points   x     n_params      x  n_dims

        return dW_dp


# noinspection PyMissingConstructor
class GlobalMDTransform(ModelDrivenTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the weights of the transform are fully specified by
    both the weights of statistical model and the weights of the
    similarity transform. The model is assumed to
    generate an instance which is then transformed by the similarity
    transform; the result defines the target landmarks of the transform.
    If no source is provided, the mean of the model is defined as the
    source landmarks of the transform.

    Parameters
    ----------
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`menpo.shape.base.PointCloud`, optional
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
    def __init__(self, model, transform_cls, global_transform, source=None):
        self.pdm = GlobalPDM(model, global_transform)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)

    def compose_after_from_vector_inplace(self, delta):
        r"""

        Composes two ModelDrivenTransforms together based on the
        first order approximation proposed by Papandreou and Maragos in [1].

        Parameters
        ----------
        delta : (N,) ndarray
            Vectorized :class:`ModelDrivenTransform` to be applied **before**
            self

        Returns
        --------
        transform : self
            self, updated to the result of the composition


        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        model_jacobian = self.pdm.model.jacobian
        points = self.pdm.model.mean.points
        n_points = self.pdm.model.mean.n_points

        # compute:
        # -> dW/dp when p=0
        # -> dW/dp when p!=0
        # -> dW/dx when p!=0 evaluated at the source landmarks

        # dW/dq when p=0 and when p!=0 are the same and given by the
        # Jacobian of the global transform evaluated at the mean of the
        # model
        dW_dq = self._global_transform_jacobian(points)
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
        dW_dS = self.pdm.global_transform.jacobian_points(points)
        dW_db = np.einsum('ilj, idj -> idj', dW_dS, dW_db_0)
        # dW_dS:  n_points  x      n_dims       x  n_dims
        # dW_db:  n_points  x     n_weights     x  n_dims

        # dW/dp is simply the concatenation of dX_dq with dX_db
        dW_dp = np.hstack((dW_dq, dW_db))
        # dW_dp:    n_points  x     n_params     x  n_dims

        dW_dx = self.transform.jacobian_points(points)
        #dW_dx = np.dot(dW_dx, self.global_transform.linear_component.T)
        # dW_dx:  n_points  x  n_dims  x  n_dims

        #TODO: Can we do this without splitting across the two dimensions?
        dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        dW_dp_0_mat = np.reshape(dW_dp_0, (n_points * self.n_dims,
                                           self.n_parameters))
        dW_dx_dW_dp_0 = dW_dp_0_mat * dW_dx_x + dW_dp_0_mat * dW_dx_y
        dW_dx_dW_dp_0 = np.reshape(dW_dx_dW_dp_0,
                                   (n_points, self.n_parameters, self.n_dims))
        # dW_dx:          n_points  x  n_dims    x  n_dims
        # dW_dp_0:        n_points  x  n_params  x  n_dims
        # dW_dx_dW_dp_0:  n_points  x  n_params  x  n_dims

        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)

        Jp = np.linalg.solve(H, J)
        # Jp:  n_params  x  n_params

        self.from_vector_inplace(self.as_vector() + np.dot(Jp, delta))

    def _global_transform_jacobian(self, points):
        return self.pdm.global_transform.jacobian(points)

    def jacobian(self, points):
        """
        Calculates the Jacobian of the ModelDrivenTransform wrt to
        its weights (the weights). This is done by chaining the relative
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

        model_jacobian = self.pdm.model.jacobian
        points = self.pdm.model.mean.points

        # compute dX/dp

        # dX/dq is the Jacobian of the global transform evaluated at the
        # mean of the model.
        dX_dq = self._global_transform_jacobian(points)
        # dX_dq:  n_points  x  n_global_params  x  n_dims

        # by application of the chain rule dX_db is the Jacobian of the
        # model transformed by the linear component of the global transform
        dS_db = model_jacobian
        dX_dS = self.pdm.global_transform.jacobian_points(points)
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


# noinspection PyMissingConstructor
class OrthoMDTransform(GlobalMDTransform):
    r"""
    A transform that couples an alignment transform to a
    statistical model together with a global similarity transform,
    such that the weights of the transform are fully specified by
    both the weights of statistical model and the weights of the
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
    model : :class:`menpo.model.base.StatisticalModel`
        A linear statistical shape model.
    transform_cls : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The align constructor will be called on this with the source
        and target landmarks. The target is
        set to the points generated from the model using the
        provide weights - the source is either given or set to the
        model's mean.
    global_transform : :class:`menpo.transform.AlignableTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`menpo.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no ``source`` is provided the
        mean of the model is used.
    """
    def __init__(self, model, transform_cls, global_transform, source=None):
        self.pdm = OrthoPDM(model, global_transform)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)

    def _global_transform_jacobian(self, points):
            return self.pdm.similarity_model.jacobian
