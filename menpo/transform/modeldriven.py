import numpy as np

from menpo.base import Targetable, Vectorizable, DP
from menpo.model.modelinstance import PDM, GlobalPDM, OrthoPDM

from .base import Transform, VComposable, VInvertible


class ModelDrivenTransform(Transform, Targetable, Vectorizable,
                           VComposable, VInvertible, DP):
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
        self._cached_points, self.dW_dl = None, None
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
            Passed through to transforms `apply_inplace` method.

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

    def _as_vector(self):
        r"""
        Return the current weights of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (`n_parameters`,) ndarray
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
        # the incremental warp is always evaluated at p=0, ie the mean shape
        points = self.pdm.model.mean.points

        # compute:
        #   - dW/dp when p=0
        #   - dW/dp when p!=0
        #   - dW/dx when p!=0 evaluated at the source landmarks

        # dW/dp when p=0 and when p!=0 are the same and simply given by
        # the Jacobian of the model
        # (n_points, n_params, n_dims)
        dW_dp_0 = self.pdm.d_dp(points)
        # (n_points, n_params, n_dims)
        dW_dp = dW_dp_0

        # (n_points, n_dims, n_dims)
        dW_dx = self.transform.d_dx(points)

        # (n_points, n_params, n_dims)
        dW_dx_dW_dp_0 = np.einsum('ijk, ilk -> eilk', dW_dx, dW_dp_0)

        #TODO: Can we do this without splitting across the two dimensions?
        # dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        # dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        # dW_dp_0_mat = np.reshape(dW_dp_0, (n_points * self.n_dims,
        #                                    self.n_parameters))
        # dW_dx_dW_dp_0 = dW_dp_0_mat * dW_dx_x + dW_dp_0_mat * dW_dx_y
        # dW_dx_dW_dp_0 = np.reshape(dW_dx_dW_dp_0,
        #                            (n_points, self.n_parameters, self.n_dims))

        # (n_params, n_params)
        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        # (n_params, n_params)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)
        # (n_params, n_params)
        Jp = np.linalg.solve(H, J)
        lsadl;saj;ls

        self.from_vector_inplace(self.as_vector() + np.dot(Jp, delta))

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

    def d_dp(self, points):
        r"""
        The derivative of this MDT wrt parametrization changes evaluated at
        points.

        This is done by chaining the derivative of points wrt the
        source landmarks on the transform (dW/dL) together with the Jacobian
        of the linear model wrt its weights (dX/dp).

        Parameters
        ----------

        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------

        ndarray shape (n_points, n_params, n_dims)
            The jacobian wrt parameterization

        """
        # check if re-computation of dW/dl can be avoided
        if not np.array_equal(self._cached_points, points):
            # recompute dW/dl, the derivative each point wrt
            # the source landmarks
            self.dW_dl = self.transform.d_dl(points)
            # cache points
            self._cached_points = points

        # dX/dp is simply the Jacobian of the PDM
        dX_dp = self.pdm.d_dp(points)

        # PREVIOUS
        # dW_dX:  n_points x n_centres x n_dims
        # dX_dp:  n_centres x n_params x n_dims

        # dW_dl:  n_points x (n_dims) x n_centres x n_dims
        # dX_dp:  (n_points x n_dims) x n_params
        dW_dp = np.einsum('ild, lpd -> ipd', self.dW_dl, dX_dp)
        # dW_dp:  n_points x n_params x n_dims

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
        The source landmarks of the transform. If no `source` is provided the
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
        # the incremental warp is always evaluated at p=0, ie the mean shape
        points = self.pdm.model.mean.points

        # compute:
        #   - dW/dp when p=0
        #   - dW/dp when p!=0
        #   - dW/dx when p!=0 evaluated at the source landmarks

        # dW/dq when p=0 and when p!=0 are the same and given by the
        # Jacobian of the global transform evaluated at the mean of the
        # model
        # (n_points, n_global_params, n_dims)
        dW_dq = self.pdm._global_transform_d_dp(points)

        # dW/db when p=0, is the Jacobian of the model
        # (n_points, n_weights, n_dims)
        dW_db_0 = PDM.d_dp(self.pdm, points)

        # dW/dp when p=0, is simply the concatenation of the previous
        # two terms
        # (n_points, n_params, n_dims)
        dW_dp_0 = np.hstack((dW_dq, dW_db_0))

        # by application of the chain rule dW_db when p!=0,
        # is the Jacobian of the global transform wrt the points times
        # the Jacobian of the model: dX(S)/db = dX/dS *  dS/db
        # (n_points, n_dims, n_dims)
        dW_dS = self.pdm.global_transform.d_dx(points)
        # (n_points, n_weights, n_dims)
        dW_db = np.einsum('ilj, idj -> idj', dW_dS, dW_db_0)

        # dW/dp is simply the concatenation of dW_dq with dW_db
        # (n_points, n_params, n_dims)
        dW_dp = np.hstack((dW_dq, dW_db))

        # dW/dx is the jacobian of the transform evaluated at the source
        # landmarks
        # (n_points, n_dims, n_dims)
        dW_dx = self.transform.d_dx(points)

        # (n_points, n_params, n_dims)
        dW_dx_dW_dp_0 = np.einsum('ijk, ilk -> ilk', dW_dx, dW_dp_0)

        #TODO: Can we do this without splitting across the two dimensions?
        # dW_dx_x = dW_dx[:, 0, :].flatten()[..., None]
        # dW_dx_y = dW_dx[:, 1, :].flatten()[..., None]
        # dW_dp_0_mat = np.reshape(dW_dp_0, (n_points * self.n_dims,
        #                                    self.n_parameters))
        # dW_dx_dW_dp_0 = dW_dp_0_mat * dW_dx_x + dW_dp_0_mat * dW_dx_y
        # # (n_points, n_params, n_dims)
        # dW_dx_dW_dp_0 = np.reshape(dW_dx_dW_dp_0,
        #                            (n_points, self.n_parameters, self.n_dims))

        # (n_params, n_params)
        J = np.einsum('ijk, ilk -> jl', dW_dp, dW_dx_dW_dp_0)
        # (n_params, n_params)
        H = np.einsum('ijk, ilk -> jl', dW_dp, dW_dp)
        # (n_params, n_params)
        Jp = np.linalg.solve(H, J)

        self.from_vector_inplace(self.as_vector() + np.dot(Jp, delta))


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
    global_transform : :class:`menpo.transform.Aligna
    bleTransform`
        A class of :class:`menpo.transform.base.AlignableTransform`
        The global transform that should be applied to the model output.
        Doesn't have to have been constructed from the .align() constructor.
        Note that the GlobalMDTransform isn't guaranteed to hold on to the
        exact object passed in here - so don't expect external changes to
        the global_transform to be reflected in the behavior of this object.
    source : :class:`menpo.shape.base.PointCloud`, optional
        The source landmarks of the transform. If no `source` is provided the
        mean of the model is used.
    """
    def __init__(self, model, transform_cls, global_transform, source=None):
        self.pdm = OrthoPDM(model, global_transform)
        self._cached_points = None
        self.transform = transform_cls(source, self.target)

