import numpy as np
from menpo.base import Copyable


class LinearModel(Copyable):
    r"""
    A Linear Model contains a matrix of vector components, each component
    vector being made up of `features`.

    Parameters
    ----------
    components : ``(n_components, n_features)`` `ndarray`
        The components array.
    """

    def __init__(self, components):
        self._components = components  # getter/setter variable

    @property
    def n_components(self):
        r"""
        The number of bases of the model.

        :type: `int`
        """
        return self._components.shape[0]

    @property
    def n_features(self):
        r"""
        The number of elements in each linear component.

        :type: `int`
        """
        return self.components.shape[-1]

    @property
    def components(self):
        r"""
        The components matrix of the linear model.

        :type: ``(n_available_components, n_features)`` `ndarray`
        """
        return self._components

    @components.setter
    def components(self, value):
        r"""
        Updates the components of this linear model, ensuring that the shape
        of the components is not changed.

        Parameters
        ----------
        value : ``(n_components, n_features)`` `ndarray`
            The new components array.

        Raises
        ------
        ValueError
            Trying to replace components of shape {} with some of shape {}
        """
        if value.shape != self._components.shape:
            raise ValueError(
                "Trying to replace components of shape {} with some of "
                "shape {}".format(self.components.shape, value.shape))
        else:
            np.copyto(self._components, value, casting='safe')

    def component_vector(self, index):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : `int`
            The component that is to be returned.

        Returns
        -------
        component_vector : ``(n_features,)`` `ndarray`
            The component vector.
        """
        return self.components[index]

    def instance_vector(self, weights):
        r"""
        Creates a new vector instance of the model by weighting together the
        components.

        Parameters
        ----------
        weights : ``(n_weights,)`` `ndarray` or `list`
            The weightings for the first `n_weights` components that should be
            used.

            ``weights[j]`` is the linear contribution of the j'th principal
            component to the instance vector.

        Returns
        -------
        vector : ``(n_features,)`` `ndarray`
            The instance vector for the weighting provided.
        """
        # just call the plural version and adapt
        weights = np.asarray(weights)  # if eg a list is provided
        return self.instance_vectors(weights[None, :]).flatten()

    def instance_vectors(self, weights):
        """
        Creates new vectorized instances of the model using all the components
        of the linear model.

        Parameters
        ----------
        weights : ``(n_vectors, n_weights)`` `ndarray` or `list` of `lists`
            The weightings for all components of the linear model. All
            components will be used to produce the instance.

            ``weights[i, j]`` is the linear contribution of the j'th
            principal component to the i'th instance vector produced.

        Raises
        ------
        ValueError
            If n_weights > n_available_components

        Returns
        -------
        vectors : ``(n_vectors, n_features)`` `ndarray`
            The instance vectors for the weighting provided.
        """
        weights = np.asarray(weights)  # if eg a list is provided
        n_instances, n_weights = weights.shape
        if not n_weights == self.n_components:
            raise ValueError(
                "Number of weightings has to match number of available "
                "components = {}".format(self.n_components))
        return self._instance_vectors_for_full_weights(weights)

    # TODO check this is right
    def _instance_vectors_for_full_weights(self, full_weights):
        return np.dot(full_weights, self.components)

    def project_vector(self, vector):
        """
        Projects the `vector` onto the model, retrieving the optimal
        linear reconstruction weights.

        Parameters
        ----------
        vector : ``(n_features,)`` `ndarray`
            A vectorized novel instance.

        Returns
        -------
        weights : ``(n_components,)`` `ndarray`
            A vector of optimal linear weights.
        """
        return self.project_vectors(vector[None, :]).flatten()

    def project_vectors(self, vectors):
        """
        Projects each of the `vectors` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vectors : ``(n_samples, n_features)`` `ndarray`
            Array of vectorized novel instances.

        Returns
        -------
        weights : ``(n_samples, n_components)`` `ndarray`
            The matrix of optimal linear weights.
        """
        return np.dot(vectors, self.components.T)

    def reconstruct_vector(self, vector):
        """
        Project a `vector` onto the linear space and rebuild from the weights
        found.

        Parameters
        ----------
        vector : ``(n_features, )`` `ndarray`
            A vectorized novel instance to project.

        Returns
        -------
        reconstructed : ``(n_features,)`` `ndarray`
            The reconstructed vector.
        """
        return self.reconstruct_vectors(vector[None, :]).flatten()

    def reconstruct_vectors(self, vectors):
        """
        Projects the `vectors` onto the linear space and rebuilds vectors from
        the weights found.

        Parameters
        ----------
        vectors : ``(n_vectors, n_features)`` `ndarray`
            A set of vectors to project.

        Returns
        -------
        reconstructed : ``(n_vectors, n_features)`` `ndarray`
            The reconstructed vectors.
        """
        return self.instance_vectors(self.project_vectors(vectors))

    def project_out_vector(self, vector):
        """
        Returns a version of `vector` where all the basis of the model have
        been projected out.

        Parameters
        ----------
        vector : ``(n_features,)`` `ndarray`
            A novel vector.

        Returns
        -------
        projected_out : ``(n_features,)`` `ndarray`
            A copy of `vector` with all basis of the model projected out.
        """
        return self.project_out_vectors(vector[None, :])

    def project_out_vectors(self, vectors):
        """
        Returns a version of `vectors` where all the basis of the model have
        been projected out.

        Parameters
        ----------
        vectors : ``(n_vectors, n_features)`` `ndarray`
            A matrix of novel vectors.

        Returns
        -------
        projected_out : ``(n_vectors, n_features)`` `ndarray`
            A copy of `vectors` with all basis of the model projected out.
        """
        weights = np.dot(vectors, self.components.T)
        return vectors - np.dot(weights, self.components)

    def orthonormalize_inplace(self):
        r"""
        Enforces that this model's components are orthonormalized,
        s.t. ``component_vector(i).dot(component_vector(j) = dirac_delta``.
        """
        Q = np.linalg.qr(self.components.T)[0].T
        self.components[...] = Q

    # TODO: Investigate the meaning and consequences of trying to
    # orthonormalize two identical vectors
    def orthonormalize_against_inplace(self, linear_model):
        r"""
        Enforces that the union of this model's components and another are
        both mutually orthonormal.

        Both models keep its number of components unchanged or else a value
        error is raised.

        Parameters
        ----------
        linear_model : :class:`LinearModel`
            A second linear model to orthonormalize this against.

        Raises
        ------
        ValueError
            The number of features must be greater or equal than the sum of the
            number of components in both linear models ({} < {})
        """
        n_components_sum = self.n_components + linear_model.n_components
        if not self.n_features >= n_components_sum:
            raise ValueError(
                "The number of features must be greater or equal than the "
                "sum of the number of components in both linear models ({} < "
                "{})".format(self.n_features, n_components_sum))
        # take the QR decomposition of the model components
        Q = (np.linalg.qr(np.hstack((linear_model._components.T,
                                     self._components.T)))[0]).T
        # set the orthonormalized components of the model being passed
        linear_model.components = Q[:linear_model.n_components, :]
        # set the orthonormalized components of this model
        self.components = Q[linear_model.n_components:, :]


class MeanLinearModel(LinearModel):
    r"""
    A Linear Model containing a matrix of vector components, each component
    vector being made up of `features`. The model additionally has a mean
    component which is handled accordingly when either:

    1. A component of the model is selected
    2. A projection operation is performed

    Parameters
    ----------
    components : ``(n_components, n_features)`` `ndarray`
        The components array.
    mean_vector : ``(n_features,)`` `ndarray`
        The mean vector.
    """
    def __init__(self, components, mean_vector):
        super(MeanLinearModel, self).__init__(components)
        self.mean_vector = mean_vector

    def component_vector(self, index, with_mean=True, scale=1.0):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : `int`
            The component that is to be returned
        with_mean : `bool`, optional
            If ``True``, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.
        scale : `float`, optional
            A scale factor that should be directly applied to the component.
            Only valid in the case where ``with_mean == True``.

        Returns
        -------
        component_vector : ``(n_features,)`` `ndarray`
            The component vector.
        """
        if with_mean:
            return (scale * self.components[index]) + self.mean_vector
        else:
            return self.components[index]

    def project_vectors(self, vectors):
        """
        Projects each of the `vectors` onto the model, retrieving
        the optimal linear reconstruction weights for each instance.

        Parameters
        ----------
        vectors : ``(n_samples, n_features)`` `ndarray`
            Array of vectorized novel instances.

        Returns
        -------
        projected : ``(n_samples, n_components)`` `ndarray`
            The matrix of optimal linear weights.
        """
        X = vectors - self.mean_vector
        return np.dot(X, self.components.T)

    def _instance_vectors_for_full_weights(self, full_weights):
        x = LinearModel._instance_vectors_for_full_weights(self, full_weights)
        return x + self.mean_vector
