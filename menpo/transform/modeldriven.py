import numpy as np
from menpo.base import Vectorizable
from menpo.transform.pdm import (PDM, GlobalPDM,
                                 OrthoPDM)

from menpo.base import VectorizableUpdatable, Targetable
from menpo.model import Similarity2dInstanceModel
from .base import Transform, VInvertible

class ModelDrivenTransform(Transform, VectorizableUpdatable, VInvertible,
                           Targetable):
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

    def composes_inplace_with(self):
        return ModelDrivenTransform

    #TODO: Rethink this transform so it knows how to deal with complex shapes
    def __init__(self, model, transform, source=None, weights=None,
                 composition='both'):
        super(ModelDrivenTransform, self).__init__()
        self.pdm_transform = PDM(model, weights=weights)
        if source is None:
            source = self.pdm_transform.source
        self._source = source
        self._cached_points = None
        self.composition = composition
        self.transform = transform.align(self.source, self.target)

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.pdm_transform.n_dims

    @property
    def n_parameters(self):
        r"""
        The total number of parameters.

        Simply ``n_weights``.

        :type: int
        """
        return self.pdm_transform.n_parameters

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: int
        """
        return self.pdm_transform.n_weights

    @property
    def has_true_inverse(self):
        return False

    def _build_pseudoinverse(self):
        return self.from_vector(-self.as_vector())

    @property
    def weights(self):
        return self.pdm_transform.weights

    @weights.setter
    def weights(self, value):
        r"""
        Setting the weights value automatically triggers a recalculation of
        the target, and an update of the transform
        """
        self.pdm_transform.weights = value
        self.transform.target = self.target

    @Alignable.target.getter
    def target(self):
        return self.pdm_transform.target

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
        dX_dp = self.pdm_transform.model.jacobian

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
        Return the current weights of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (``n_parameters``,) ndarray
            The vector of weights
        """
        return self.pdm_transform.as_vector()

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
        self.pdm_transform.target = new_target
        self.transform.target = self.target

    def _sync_state_from_target(self):
        pass

    @property
    def target(self):
        pass


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

    def update_from_vector_inplace(self, delta):
        r"""

        compose_after this :class:`ModelDrivenTransform` with another inplace.
        Rather than requiring a new ModelDrivenTransform to compose_after
        with, this method only requires the weights of the new transform.

        Parameters
        ----------
        delta : (N,) ndarray
            Vectorized :class:`ModelDrivenTransform` to be applied **before**
            self

        Returns
        --------
        transform : self
            self, updated to the result of the composition
        """
        if self.composition is 'model':
            new_mdtransform = self.from_vector(delta)
            self.target = self._compose_after_model(new_mdtransform.target)
        elif self.composition is 'warp':
            new_mdtransform = self.from_vector(delta)
            self.target = self._compose_after_warp(new_mdtransform.target)
        elif self.composition is 'both':
            self.from_vector_inplace(self._compose_after_both(delta))
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
        from menpo.shape import PointCloud
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
        first order approximation proposed by Papandreou and Maragos in [1].

        The resulting vector of weights is equivalent to

            self.compose_after_from_vector(mdt_vector)

        Parameters
        ----------
        mdt_vector : (P,) ndarray
            the weights of the ModelDrivenTransform we are
            composing with, as provided by .as_vector().

        Returns
        -------
        vector: (P,) ndarray
            The new weights of the composed result

        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        model_jacobian = self.pdm_transform.model.jacobian
        points = self.pdm_transform.model.mean.points
        n_points = self.pdm_transform.model.mean.n_points

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

        return self.as_vector() + np.dot(Jp, mdt_vector)

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
        return self.pdm_transform.pseudoinverse_vector(vector)


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
    def __init__(self, model, transform, global_transform, source=None,
                 weights=None, composition='both'):
        super(ModelDrivenTransform, self).__init__()
        self.pdm_transform = GlobalPDM(model, global_transform,
                                                weights=weights)
        if source is None:
            source = self.pdm_transform.source
        self._source = source
        self._cached_points = None
        self.composition = composition
        self.transform = transform.align(self.source, self.target)

    @property
    def n_global_parameters(self):
        r"""
        The number of weights in the ``global_transform``

        :type: int
        """
        return self.pdm_transform.n_global_parameters

    @property
    def global_parameters(self):
        r"""
        The weights for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.pdm_transform.global_parameters

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

        model_jacobian = self.pdm_transform.model.jacobian
        points = self.pdm_transform.model.mean.points

        # compute dX/dp

        # dX/dq is the Jacobian of the global transform evaluated at the
        # mean of the model.
        dX_dq = self._global_transform_jacobian(points)
        # dX_dq:  n_points  x  n_global_params  x  n_dims

        # by application of the chain rule dX_db is the Jacobian of the
        # model transformed by the linear component of the global transform
        dS_db = model_jacobian
        dX_dS = self.pdm_transform.global_transform.jacobian_points(points)
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
        return self.pdm_transform.global_transform.jacobian(points)

    def from_vector_inplace(self, vector):
        self.pdm_transform.from_vector_inplace(vector)
        self.transform.target = self.target

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
        from menpo.shape import PointCloud
        return PointCloud(composed_target)

    def _compose_after_both(self, mdt_vector):
        r"""
        Composes two statistically driven transforms together based on the
        first order approximation proposed by Papandreou and Maragos.

        Parameters
        ----------
        new_sdt_parameters : (P,) ndarray
            the weights of the ModelDrivenTransform we are
            composing with, as provided by .as_vector().

        Returns
        -------

        weights: (P,) ndarray
            The new weights of the composed result

        References
        ----------

        .. [1] G. Papandreou and P. Maragos, "Adaptive and Constrained
               Algorithms for Inverse Compositional Active Appearance Model
               Fitting", CVPR08
        """
        model_jacobian = self.pdm_transform.model.jacobian
        points = self.pdm_transform.model.mean.points
        n_points = self.pdm_transform.model.mean.n_points

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
        dW_dS = self.pdm_transform.global_transform.jacobian_points(points)
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

        return self.as_vector() + np.dot(Jp, mdt_vector)


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
    weights : (P,) ndarray, optional
        The reconstruction weights that will be fed to the model in order to
        generate an instance of the target landmarks.
    composition: 'both', 'warp' or 'model', optional
        The composition approximation employed by this
        ModelDrivenTransform.

        Default: `both`
    """
    def __init__(self, model, transform, global_transform, source=None,
                 weights=None, composition='both'):
        super(ModelDrivenTransform, self).__init__()
        self.pdm_transform = OrthoPDM(model, global_transform,
                                               weights=weights)
        if source is None:
            source = self.pdm_transform.source
        self._source = source
        self._cached_points = None
        self.composition = composition
        self.transform = transform.align(self.source, self.target)

    def _global_transform_jacobian(self, points):
        return self.pdm_transform.similarity_model.jacobian
