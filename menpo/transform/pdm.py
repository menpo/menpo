from copy import deepcopy
import numpy as np
from menpo.model import Similarity2dInstanceModel
from menpo.transform.base import Alignable, ComposableTransform


# TODO: document me
class PDMTransform(ComposableTransform, Alignable):
    r"""
    """
    #TODO: Rethink this transform so it knows how to deal with complex shapes
    def __init__(self, model, weights=None):
        self.model = model
        # TODO: this is just done here to be able to use the Alignable
        # interface, need to rethink this whole transform probably.
        self._source = model.mean

        if weights is None:
            # set all weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_active_components)
        self._weights = weights

        self._target = self._target_for_weights(self._weights)

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.model.template_instance.n_dims

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
        return True

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

    # TODO: document me
    def jacobian(self, points):
        """
        """
        return self.model.jacobian

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

    # TODO: document me
    def _target_setter(self, new_target):
        r"""
        """
        # 1. Find the optimum weights and set them
        self._weights = self._weights_for_target(new_target)
        # 2. Find the closest target the model can reproduce and trigger an
        # update of our transform
        self._target = self._target_for_weights(self._weights)

    # TODO: document me
    def _apply(self, **kwargs):
        r"""
        """
        return self.target

    def composes_inplace_with(self):
        return PDMTransform

    def _compose_before_inplace(self, transform):
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

    # TODO: document me
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
        self_copy._compose_after_inplace(transform)
        return self_copy

    # TODO: document me
    def _compose_after_inplace(self, pdm_transform):
        r"""
        """
        new_params = self.compose_after_from_vector_inplace(
            pdm_transform.as_vector())
        return self.from_vector_inplace(new_params)

    # TODO: document me
    def compose_after_from_vector_inplace(self, vector):
        r"""
        """
        self.weights = self.weights + vector
        return self

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

        new_target: :class:`menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.model.instance(weights)

    def _weights_for_target(self, target):
        r"""
        Return the appropriate model weights for target provided.
        Subclasses can override this.

        Parameters
        ----------

        target: :class:`menpo.shape.PointCloud`
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


# TODO: document me
class GlobalPDMTransform(PDMTransform):
    r"""
    """
    def __init__(self, model, global_transform, weights=None):
        # need to set the global transform right away - self
        # ._target_for_weights() needs it in superclass __init__
        self.global_transform = global_transform
        super(GlobalPDMTransform, self).__init__(model, weights=weights)
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
        r"""
        """
        global_transform_jacobian = self._global_transform_jacobian(
            self.model.mean.points)
        return np.hstack((global_transform_jacobian.T,
                          self.model.components.T))

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
        model_weights = vector[self.n_global_parameters:]
        self._update_global_weights(global_params)
        self.weights = model_weights

    # TODO: document me
    def compose_after_from_vector_inplace(self, vector):
        r"""
        """
        global_params = (self.global_parameters +
                         vector[:self.n_global_parameters])
        model_weights = (self.weights +
                         vector[self.n_global_parameters:])
        self._update_global_weights(global_params)
        self.weights = model_weights
        return self

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform.from_vector_inplace(global_weights)

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

        new_target: :class:`menpo.shape.PointCloud`
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

        target: :class:`menpo.shape.PointCloud`
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


# TODO: document me
class OrthoPDMTransform(GlobalPDMTransform):
    r"""
    """
    def __init__(self, model, global_transform, weights=None):
        # 1. Construct similarity model from the mean of the model
        self.similarity_model = Similarity2dInstanceModel(model.mean)
        # 2. Orthonormalize model and similarity model
        model = deepcopy(model)
        model.orthonormalize_against_inplace(self.similarity_model)
        self.similarity_weights = self.similarity_model.project(
            global_transform.apply(model.mean))

        super(OrthoPDMTransform, self).__init__(
            model, global_transform, weights=weights)

    def _update_global_transform(self, target):
        self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.target = new_target

    def _global_transform_jacobian(self, points):
        return self.similarity_model.components

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.similarity_weights
