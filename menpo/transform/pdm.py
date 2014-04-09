from copy import deepcopy
import numpy as np
from menpo.base import Targetable, VectorizableUpdatable
from menpo.model import Similarity2dInstanceModel
from .base import VInvertible


# TODO: document me
class PDM(Targetable, VectorizableUpdatable, VInvertible):
    r"""
    """
    def __init__(self, model, weights=None):
        self.model = model
        if weights is None:
            # set all weights to 0 (yielding the mean)
            weights = np.zeros(self.model.n_active_components)
        self._weights = weights
        # cannot really call set_target since the verification is triggered
        # and no target has been assigned yet... One option here would be to
        # assign self.target to model.mean and then call set_target (instead
        #  of this two calls) here
        self._target_setter(self._new_target_from_state())
        self._sync_state_from_target()

    @property
    def n_dims(self):
        r"""
        The number of dimensions that the transform supports.

        :type: int
        """
        return self.model.template_instance.n_dims

    @property
    def n_weights(self):
        r"""
        The number of parameters in the linear model.

        :type: int
        """
        return self.model.n_active_components

    @property
    def weights(self):
        r"""
        In this simple PDM the weights is just the vector, but in subclasses
        this behavior will change.
        """
        return self._weights

    @property
    def target(self):
        return self._target

    def _target_setter(self, new_target):
        r"""
        Called by the Targetable framework when set_target() is called.
        This method **ONLY SETS THE NEW TARGET** it does no synchronisation
        logic (for that, see _sync_state_from_target())
        """
        self._target = new_target

    def _sync_state_from_target(self):
        # 1. Find the optimum parameters and set them
        self._weights = self._weights_for_target(self.target)
        # 2. Find the closest target the model can reproduce and trigger an
        # update of our transform
        self._target_setter(self._new_target_from_state())

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the parameters provided.
        Subclasses can override this.

        Returns
        -------

        new_target: :class:`menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.model.instance(self.weights)

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
        self.set_target(vector)

    def update_from_vector_inplace(self, delta):
        r"""
        Additively update this object with a delta vector inplace.
        """
        self.from_vector_inplace(self.as_vector() + delta)

    @property
    def has_true_inverse(self):
        return True

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
        # just have to negate the parameters!
        return -vector

    # TODO: document me
    def jacobian(self, points):
        """
        """
        return self.model.jacobian


# TODO: document me
class GlobalPDM(PDM):
    r"""
    """
    def __init__(self, model, global_transform, weights=None):
        self.global_transform = global_transform
        super(GlobalPDM, self).__init__(model, weights=weights)

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

    def _new_target_from_state(self):
        r"""
        Return the appropriate target for the model weights provided,
        accounting for the effect of the global transform


        Returns
        -------

        new_target: :class:`menpo.shape.PointCloud`
            A new target for the weights provided
        """
        return self.global_transform.apply(self.model.instance(self.weights))

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

    def as_vector(self):
        r"""
        Return the current parameters of this transform - this is the
        just the linear model's weights

        Returns
        -------
        params : (``n_parameters``,) ndarray
            The vector of parameters
        """
        return np.hstack([self.global_parameters, self.weights])

    def from_vector_inplace(self, vector):
        global_parameters = vector[:self.n_global_parameters]
        weights = vector[self.n_global_parameters:]
        self._update_global_weights(global_parameters)
        self._weights = weights
        self._sync_target_from_state()

    def _update_global_weights(self, global_weights):
        r"""
        Hook that allows for overriding behavior when the global weights are
        set. Default implementation simply asks global_transform to
        update itself from vector.
        """
        self.global_transform.from_vector_inplace(global_weights)

    def jacobian(self, points):
        r"""
        """
        return np.hstack((self.global_transform.jacobian(points).T,
                          self.model.components.T))


# TODO: document me
class OrthoPDM(GlobalPDM):
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

        super(OrthoPDM, self).__init__(
            model, global_transform, weights=weights)

    @property
    def global_parameters(self):
        r"""
        The parameters for the global transform.

        :type: (``n_global_parameters``,) ndarray
        """
        return self.similarity_weights

    def _update_global_transform(self, target):
        self.similarity_weights = self.similarity_model.project(target)
        self._update_global_weights(self.similarity_weights)

    def _update_global_weights(self, global_weights):
        self.similarity_weights = global_weights
        new_target = self.similarity_model.instance(global_weights)
        self.global_transform.target = new_target

    def jacobian(self, points):
        r"""
        """
        return np.hstack((self.similarity_model.components.T,
                          self.model.components.T))

