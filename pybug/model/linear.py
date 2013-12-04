import numpy as np
from scipy.linalg.blas import dgemm


class LinearModel(object):
    r"""
    A Linear Model contains a matrix of vector components, each component
    vector being made up of `features`.
    """

    def __init__(self, components):
        self._components = components  # getter/setter variable
        self._n_components = self.n_available_components


    @property
    def n_available_components(self):
        r"""
        The number of bases of the model

        type: int
        """
        return self._components.shape[0]

    @property
    def n_components(self):
        r"""
        The number of components currently in use on this model.
        """
        return self._n_components

    @n_components.setter
    def n_components(self, value):
        value = round(value)
        if 0 < value <= self.n_available_components:
            self._n_components = value
        else:
            raise ValueError(
                "Tried setting n_components as {} - has to be an int and "
                "0 < n_components <= n_available_components "
                "(which is {}) ".format(value, self.n_available_components))

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
            self.n_components = n_components

    def component_vector(self, index):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : int
            The component that is to be returned

        :type: (n_features,) ndarray
        """
        return self.components[index]

    def instance_vector(self, weights):
        r"""
        Creates a new vector instance of the model by weighting together the
        components.

        Parameters
        ----------
        weights : (n_weights,) ndarray or list
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
        vector : (n_features,) ndarray
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
        weights : (n_vectors, n_weights) ndarray or list of lists
            The weightings for the first n_weights components that
            should be used per instance that is to be produced

            ``weights[i, j]`` is the linear contribution of the j'th
            principal component to the i'th instance vector produced.

        Raises
        ------
        ValueError: If n_weights > n_components

        Returns
        -------
        vectors : (n_vectors, n_features) ndarray
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

    def project_vector(self, vector):
        """
        Projects the ``vector`` onto the model, retrieving the optimal
        linear reconstruction weights

        Parameters
        -----------
        vector : (n_features,) ndarray
            A vectorized novel instance.

        Returns
        -------
        weights : (n_components,)
            A vector of optimal linear weights
        """
        return self.project_vectors(vector[None, :]).flatten()

    def project_vectors(self, vectors):
        """
        Projects each of the ``vectors`` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vectors : (n_samples, n_features) ndarray

        Returns
        -------
        weights : (n_samples, n_components) ndarray
            The matrix of optimal linear weights

        """
        return dgemm(alpha=1.0, a=vectors.T, b=self.components.T,
                     trans_a=True)

    def reconstruct_vector(self, vector):
        """
        Project a ``vector`` onto the linear space and
        rebuild from the weights found.

        Parameters
        ----------
        vector : (n_features, ) ndarray
            A vectorized novel instance to project

        Returns
        -------
        reconstructed : (n_features,) ndarray
            The reconstructed vector.
        """
        return self.reconstruct_vectors(vector[None, :]).flatten()

    def reconstruct_vectors(self, vectors):
        """
        Projects the ``vectors`` onto the linear space and
        rebuilds vectors from the weights found.

        Parameters
        ----------
        vectors : (n_vectors, n_features) ndarray
            A set of vectors to project

        Returns
        -------
        reconstructed : (n_vectors, n_features) ndarray
            The reconstructed vectors.
        """
        return self.instance_vectors(self.project_vectors(vectors))

    def project_out_vector(self, vector):
        """
        Returns a version of ``vector`` where all the basis of the
        model have been projected out.

        Parameters
        ----------
        vector : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        projected_out : (n_features,) ndarray
            A copy of ``vector`` with all basis of the model
            projected out.
        """
        return self.project_out_vectors(vector[None, :])

    def project_out_vectors(self, vectors):
        """
        Returns a version of ``vectors`` where all the basis of the
        model have been projected out.

        Parameters
        ----------
        vectors : (n_vectors, n_features) ndarray
            A matrix of novel vectors.

        Returns
        -------
        projected_out : (n_vectors, n_features) ndarray
            A copy of ``vectors`` with all basis of the model
            projected out.
        """
        weights = self.project_vectors(vectors)
        return (vectors -
                dgemm(alpha=1.0, a=weights.T, b=self.components.T,
                      trans_a=True, trans_b=False))

    def orthonormalize_inplace(self):
        r"""
        Enforces that this models components are orthonormalized

        s.t. component_vector(i).dot(component_vector(j) = dirac_delta
        """
        # TODO ask Joan
        Q, r = np.linalg.qr(self.components.T).T
        self.components[...] = Q

    def orthonormalize_against_inplace(self, linear_model):
        r"""
        Enforces that the union of this model's components and another are
        both mutually orthonormal.

        Note that the model passed in is guaranteed to not have it's number
        of available components changed. This model, however, may loose some
        dimensionality due to reaching a degenerate state.

        Paramerters
        -----------
        linear_model : :class:`LinearModel`
            A second linear model to orthonormalize this against.
        """
        # take the QR decomposition of the model components
        Q = (np.linalg.qr(np.hstack((linear_model._components.T,
                                     self._components.T)))[0]).T
        # the model passed to us went first, so all it's components will
        # survive. Pull them off, and update the other model.
        linear_model.components = Q[:linear_model.n_available_components, :]
        # it's possible that all of our components didn't survive due to
        # degeneracy. We need to trim our components down before replacing
        # them to ensure the number of components is consistent (otherwise
        # the components setter will complain at us)
        self.trim_components(
            n_components=Q.shape[0] - linear_model.n_available_components)
        # now we can set our own components with the updated orthogonal ones
        self.components = Q[linear_model.n_available_components:, :]


class MeanLinearModel(LinearModel):
    r"""
    A Linear Model containing a matrix of vector components, each component
    vector being made up of `features`. The model additionally has a mean
    component which is handled accordingly when either:

    1. A component of the model is selected
    2. A projection operation is performed

    """
    def __init__(self, components, mean_vector):
        super(MeanLinearModel, self).__init__(components)
        self.mean_vector = mean_vector

    def component_vector(self, index, with_mean=True, scale=1.0):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : int
            The component that is to be returned

        with_mean: boolean (optional)
            If True, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.

            Default: True
        scale : float
            A scale factor that should be directly applied to the component.
            Only valid in the case where with_mean is True.

        :type: (n_features,) ndarray
        """
        if with_mean:
            return (scale * self.components[index]) + self.mean_vector
        else:
            return self.components[index]

    def project_vectors(self, vectors):
        """
        Projects each of the ``vectors`` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vectors : (n_samples, n_features) ndarray

        Returns
        -------
        projected: (n_samples, n_components) ndarray
            The matrix of optimal linear weights

        """
        X = vectors - self.mean_vector
        return dgemm(alpha=1.0, a=X.T, b=self.components.T, trans_a=True)

    def _instance_vectors_for_full_weights(self, full_weights):
        x = LinearModel._instance_vectors_for_full_weights(self, full_weights)
        return x + self.mean_vector
