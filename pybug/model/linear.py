import numpy as np
from scipy.linalg.blas import dgemm
from pybug.model.instancebacked import InstanceBackedModel


class LinearModel(object):
    r"""
    A Linear Model contains a matrix of vector components, each component
    vector being made up of `features`.
    """

    def __init__(self, components):
        self._components = components  # getter/setter variable
        self.n_components = self.n_available_components

    @property
    def n_available_components(self):
        r"""
        The number of bases of the model

        type: int
        """
        return self._components.shape[0]

    @property
    def n_features(self):
        r"""
        The number of elements in each linear component.

        type: int
        """
        return self.components.shape[-1]

    @property
    def components(self):
        r"""
        The component matrix of the linear model.

        type: (n_components, n_features) ndarray
        """
        return self._components[:self.n_components]

    @components.setter
    def components(self, value):
        r"""
        Updates the components of this linear model, ensuring that the shape
        of the components is not changed.
        """
        if value.shape != self._components.shape:
            raise ValueError(
                "Trying to replace components of shape {} with some of "
                "shape {}".format(self.components.shape, value.shape))
        else:
            np.copyto(self._components, value, casting='safe')

    def trim_components(self, n_components=None):
        r"""
        Permanently trims the components down to a certain amount.

        Parameters
        ----------

        n_components: int, optional
            The number of components that are kept. If None,
            self.n_components is used.
        """
        if n_components is None:
            n_components = self.n_components

        if not n_components < self.n_available_components:
            raise ValueError(
                "n_components ({}) needs to be less than "
                "n_available_components ({})".format(
                n_components, self.n_available_components))
        else:
            self._components = self._components[:n_components]

    def component_vector(self, index):
        """
        A particular component of the model, in vectorized form.

        :type: (n_features,) ndarray
        """
        return self.components[:, index]

    def instance_vector(self, weights):
        r"""
        Creates a new vector instance of the model by weighting together the
        components.

        Parameters
        ----------
        weights : (n_weights,) ndarray
            The weightings for the first n_weights components that
            should be used.

            ``weights[j]`` is the linear contribution of the j'th principal
            component to the instance vector. Note that if n_weights <
            n_components, only the first n_weight components are used in the
            reconstruction (i.e. unspecified weights are implicitly 0)

        Raises
        ------
        ValueError: If n_weights > n_components

        Returns
        -------
        instance : (n_features,) ndarray
            The instance vector for the weighting provided.
        """
        # just call the plural version and adapt
        weights = np.asarray(weights)  # if eg a list is provided
        return self.instance_vectors(weights[None, :]).flatten()

    def instance_vectors(self, weights):
        """
        Creates new vectorized instances of the model using the first
        components in a particular weighting.

        Parameters
        ----------
        weights : (n_instances, n_weights) ndarray
            The weightings for the first n_weights components that
            should be used per instance that is to be produced

            ``weights[i, j]`` is the linear contribution of the j'th
            principal component to the i'th instance vector produced.

        Raises
        ------
        ValueError: If n_weights > n_components

        Returns
        -------
        instance : (n_instances, n_features) ndarray
            The instance vectors for the weighting provided.
        """
        weights = np.asarray(weights)  # if eg a list is provided
        n_instances, n_weights = weights.shape
        if n_weights > self.n_components:
            raise ValueError(
                "Number of weightings cannot be greater than {}".format(
                    self.n_components))
        else:
            full_weights = np.zeros((n_instances, self.n_components))
            full_weights[..., :n_weights] = weights
            weights = full_weights
        return self._instance_vectors_for_full_weights(weights)

    # TODO check this is right
    def _instance_vectors_for_full_weights(self, full_weights):
        return dgemm(alpha=1.0, a=full_weights.T, b=self.components.T,
                  trans_a=True, trans_b=True)

    def project_vector(self, vector_instance):
        """
        Projects the ``vector_instance`` onto the model, retrieving the optimal
        linear reconstruction weights

        Parameters
        -----------
        vector_instance : (n_features,) ndarray
            A vectorized novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of optimal linear weights
        """
        return self.project_vectors(vector_instance[None, :]).flatten()

    def project_vectors(self, vector_instances):
        """
        Projects each of the ``vector_instances`` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vector_instances : (n_samples, n_features) ndarray

        Returns
        -------
        projected: (n_samples, n_components) ndarray
            The matrix of optimal linear weights

        """
        return dgemm(alpha=1.0, a=vector_instances.T, b=self.components.T,
                     trans_a=True)

    def reconstruct_vectors(self, instance_vectors, n_components=None):
        """
        Project a  ``novel_instance`` onto the linear space and
        rebuild from the weights found.

        Parameters
        ----------
        instance_vectors : (n_features, ) ndarray
            A vectorized novel instance to project
        n_components : int, optional
            The number of components to use in the reconstruction

            Default: ``weights.shape[0]``

        Returns
        -------
        reconstructed : (n_features,) ndarray
            The reconstructed vector.
        """
        weights = self.project_vector(instance_vectors)
        if n_components is not None:
            weights = weights[..., :n_components]
        return self.instance_vectors(weights)

    def project_out_vector(self, instance_vector):
        """
        Returns a version of ``instance_vector`` where all the basis of the
        model have been projected out.

        Parameters
        ----------
        instance_vector : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        projected_out : (n_features,) ndarray
            A copy of ``instance_vector`` with all basis of the model
            projected out.
        """
        weights = dgemm(alpha=1.0, a=instance_vector.T, b=self.components.T,
                        trans_a=True)
        return (instance_vector -
                dgemm(alpha=1.0, a=weights.T, b=self.components.T,
                      trans_a=True, trans_b=True))

    def orthonormalize_inplace(self):
        r"""
        Enforces that this models components are orthonormalized

        s.t. component_vector(i).dot(component_vector(j) = d_ij (the dirac
        delta)

        """
        # TODO ask Joan
        Q, r = np.linalg.qr(self.components.T).T
        self.components[...] = Q

    def orthonormalize_against_inplace(self, linear_model):
        #TODO document and check
        r"""
        """
        Q = (np.linalg.qr(np.hstack((linear_model._components.T,
                                     self._components.T)))[0]).T

        linear_model.components = Q[:linear_model.n_available_components, :]
        self.components = Q[linear_model.n_available_components:, :]


class MeanLinearModel(LinearModel):

    def __init__(self, components, mean_vector):
        super(MeanLinearModel, self).__init__(components)
        self.mean_vector = mean_vector

    def component_vector(self, index):
        return self.component_vector(index) + self.mean_vector

    def project_vectors(self, vector_instances):
        """
        Projects each of the ``vector_instances`` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vector_instances : (n_samples, n_features) ndarray

        Returns
        -------
        projected: (n_samples, n_components) ndarray
            The matrix of optimal linear weights

        """
        X = vector_instances - self.mean_vector
        return dgemm(alpha=1.0, a=X.T, b=self.components.T, trans_a=True)

    def _instance_vectors_for_full_weights(self, full_weights):
        x = LinearModel._instance_vectors_for_full_weights(self, full_weights)
        return x + self.mean_vector
