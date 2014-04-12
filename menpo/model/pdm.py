from copy import deepcopy
import numpy as np

from menpo.base import Targetable, Vectorizable
from menpo.model import Similarity2dInstanceModel


# TODO: document me
class PDM(Targetable, Vectorizable):
    r"""
    """
    def __init__(self, model):
        self.model = model
        self._target = None
        # set all weights to 0 (yielding the mean, first call to
        # from_vector_inplace() or set_target() will update this)
        self._weights = np.zeros(self.model.n_active_components)
        self._sync_target_from_state()

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

    def _sync_state_from_target(self):
        # 1. Find the optimum parameters and set them
        self._weights = self._weights_for_target(self.target)
        # 2. Find the closest target the model can reproduce and trigger an
        # update of our transform
        self._target_setter(self._new_target_from_state())

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
        self._weights = vector
        self._sync_target_from_state()

    # TODO: document me
    def jacobian(self, points):
        """
        """
        return self.model.jacobian


# TODO: document me
class GlobalPDM(PDM):
    r"""
    """
    def __init__(self, model, global_transform_cls):
        # Start the global_transform as an identity (first call to
        # from_vector_inplace() or set_target() will update this)
        self.global_transform = global_transform_cls(model.mean, model.mean)
        super(GlobalPDM, self).__init__(model)

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
        self.global_transform.set_target(target)

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
        # First, update the global transform
        global_parameters = vector[:self.n_global_parameters]
        self._update_global_weights(global_parameters)
        # Now extract the weights, and let super handle the update
        weights = vector[self.n_global_parameters:]
        PDM.from_vector_inplace(self, weights)

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
    def __init__(self, model, global_transform_cls):
        # 1. Construct similarity model from the mean of the model
        self.similarity_model = Similarity2dInstanceModel(model.mean)
        # 2. Orthonormalize model and similarity model
        model = deepcopy(model)
        model.orthonormalize_against_inplace(self.similarity_model)
        self.similarity_weights = self.similarity_model.project(model.mean)
        super(OrthoPDM, self).__init__(model, global_transform_cls)

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
        self.global_transform.set_target(new_target)

    def jacobian(self, points):
        r"""
        """
        return np.hstack((self.similarity_model.components.T,
                          self.model.components.T))

