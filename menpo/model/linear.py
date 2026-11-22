import numpy as np

from menpo.base import Copyable, describe

# The arrays of a model are never written to: the methods that change a model
# (the ones whose names end in _inplace) give it new arrays. So a model made by
# PCAVectorModel.trim_components(), which shares the arrays of the model it came from,
# can not change it and is not changed by it.


def _read_only(array):
    # A view of `array` that can not be written to: for handing out the memory of
    # a model (or sharing it with another model) without it being changed by
    # accident. The array itself is untouched.
    view = array.view()
    view.flags.writeable = False
    return view


def _as_components(array, like):
    # New components from a result (e.g. of a QR decomposition): C-contiguous,
    # as the components are everywhere else, and of the dtype of the old ones
    return np.ascontiguousarray(array, dtype=like.dtype)


class LinearVectorModel(Copyable):
    r"""
    A Linear Model contains a matrix of vector components, each component
    vector being made up of `features`.

    Parameters
    ----------
    components : ``(n_components, n_features)`` `ndarray`
        The components array.
    """

    def __str__(self):
        return describe(
            self, {"n_components": self.n_components, "n_features": self.n_features}
        )

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
        The components matrix of the linear model. This is a read-only view
        of the memory of the model, not a copy: it can not be written to (to
        change the components see :meth:`orthonormalize_inplace`), and it shows
        any change that is made to the model.

        :type: ``(n_available_components, n_features)`` `ndarray`
        """
        return _read_only(self._components)

    def component(self, index):
        r"""
        A particular component of the model.

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

    def instance(self, weights):
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
                "components = {}".format(self.n_components)
            )
        return self._instance_vectors_for_full_weights(weights)

    def _instance_vectors_for_full_weights(self, full_weights):
        return np.dot(full_weights, self.components)

    def project(self, vector):
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

    def reconstruct(self, vector):
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

    def _check_impute_args(self, vectors, mask):
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (self.n_features,):
            raise ValueError(
                "mask must be a boolean array of shape (n_features,) = "
                "({},), got dtype {} and shape {}".format(
                    self.n_features, mask.dtype, mask.shape
                )
            )
        vectors = np.asarray(vectors)
        n_available = int(mask.sum())
        if vectors.ndim != 2 or vectors.shape[1] != n_available:
            raise ValueError(
                "vectors must have shape (n_vectors, n_available_features) = "
                "(n_vectors, {}) where mask is True, got {}".format(
                    n_available, vectors.shape
                )
            )
        return vectors, mask

    def impute(self, vector, mask):
        """
        Imputes the features of `vector` that are missing: the weights are
        found from the features that are available, and the whole vector is
        rebuilt from them.

        The missing features are taken to be zero (the mean, for a model with
        a mean), so this is the same as filling them in with that value and
        calling :meth:`reconstruct`. It is a simple method: the weights are
        biased towards zero by the missing features.

        Parameters
        ----------
        vector : ``(n_available_features,)`` `ndarray`
            The available features of a vector, in the order of the ``True``
            entries of `mask`.
        mask : ``(n_features,)`` `bool ndarray`
            ``True`` where a feature is available in `vector`.

        Returns
        -------
        imputed : ``(n_features,)`` `ndarray`
            The imputed vector.

        Raises
        ------
        ValueError
            If `mask` is not a boolean array of length ``n_features``, or
            `vector` does not have as many features as `mask` has ``True``
            entries.
        """
        return self.impute_vectors(np.asarray(vector)[None, :], mask).flatten()

    def impute_vectors(self, vectors, mask):
        """
        Imputes the missing features of each of the `vectors`. See
        :meth:`impute`.

        Parameters
        ----------
        vectors : ``(n_vectors, n_available_features)`` `ndarray`
            The available features of each vector, in the order of the
            ``True`` entries of `mask`.
        mask : ``(n_features,)`` `bool ndarray`
            ``True`` where a feature is available in `vectors`.

        Returns
        -------
        imputed : ``(n_vectors, n_features)`` `ndarray`
            The imputed vectors.

        Raises
        ------
        ValueError
            If `mask` is not a boolean array of length ``n_features``, or
            `vectors` does not have as many columns as `mask` has ``True``
            entries.
        """
        vectors, mask = self._check_impute_args(vectors, mask)
        weights = np.dot(vectors, self.components[:, mask].T)
        return self.instance_vectors(weights)

    def project_out(self, vector):
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
        weights = self.project_vectors(vectors)
        return vectors - self._instance_vectors_for_full_weights(weights)

    def orthonormalize_inplace(self):
        r"""
        Enforces that this model's components are orthonormalized,
        s.t. ``component_vector(i).dot(component_vector(j) = dirac_delta``.

        This model gets new components; the array it had is not written to.
        The QR decomposition needs several times the memory of the components
        while it runs.
        """
        Q = np.linalg.qr(self.components.T)[0].T
        self._components = _as_components(Q, like=self._components)

    # TODO: Investigate the meaning and consequences of trying to
    # orthonormalize two identical vectors
    def orthonormalize_against_inplace(self, linear_model):
        r"""
        Enforces that the union of this model's components and another are
        both mutually orthonormal.

        Both models keep its number of components unchanged or else a value
        error is raised.

        Both models get new components; the arrays they had are not written to.

        Parameters
        ----------
        linear_model : :class:`LinearVectorModel`
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
                "{})".format(self.n_features, n_components_sum)
            )
        # take the QR decomposition of the model components
        Q = (
            np.linalg.qr(np.hstack((linear_model._components.T, self._components.T)))[0]
        ).T
        # set the orthonormalized components of the model being passed
        linear_model._components = _as_components(
            Q[: linear_model.n_components, :], like=linear_model._components
        )
        # set the orthonormalized components of this model
        self._components = _as_components(
            Q[linear_model.n_components :, :], like=self._components
        )


class MeanLinearVectorModel(LinearVectorModel):
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
    mean : ``(n_features,)`` `ndarray`
        The mean vector.
    """

    def __init__(self, components, mean):
        super(MeanLinearVectorModel, self).__init__(components)
        self._mean = mean

    def mean(self):
        r"""
        Return the mean of the model: a read-only view of the memory of the
        model, not a copy.

        :type: `ndarray`
        """
        return _read_only(self._mean)

    def component(self, index, with_mean=True, scale=1.0):
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
            return (scale * self.components[index]) + self._mean
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
        X = vectors - self._mean
        return np.dot(X, self.components.T)

    def impute_vectors(self, vectors, mask):
        """
        Imputes the missing features of each of the `vectors`. The missing
        features are taken to be the mean, so this is the same as filling them
        in with the mean and calling :meth:`reconstruct_vectors`. See
        :meth:`LinearVectorModel.impute`.

        Parameters
        ----------
        vectors : ``(n_vectors, n_available_features)`` `ndarray`
            The available features of each vector, in the order of the
            ``True`` entries of `mask`.
        mask : ``(n_features,)`` `bool ndarray`
            ``True`` where a feature is available in `vectors`.

        Returns
        -------
        imputed : ``(n_vectors, n_features)`` `ndarray`
            The imputed vectors.

        Raises
        ------
        ValueError
            If `mask` is not a boolean array of length ``n_features``, or
            `vectors` does not have as many columns as `mask` has ``True``
            entries.
        """
        vectors, mask = self._check_impute_args(vectors, mask)
        weights = np.dot(vectors - self._mean[mask], self.components[:, mask].T)
        return self.instance_vectors(weights)

    def project_out_vectors(self, vectors):
        """
        Returns a version of `vectors` where all the bases of the model have
        been projected out.

        Parameters
        ----------
        vectors : ``(n_vectors, n_features)`` `ndarray`
            A matrix of novel vectors.

        Returns
        -------
        projected_out : ``(n_vectors, n_features)`` `ndarray`
            A copy of `vectors` with all bases of the model projected out.
        """
        weights = self.project_vectors(vectors)
        # We don't add the mean back, in fact the residual is defined as
        # the mean subtracted.
        return (
            vectors - self._mean[None, ...]
        ) - LinearVectorModel._instance_vectors_for_full_weights(self, weights)

    def _instance_vectors_for_full_weights(self, full_weights):
        x = LinearVectorModel._instance_vectors_for_full_weights(self, full_weights)
        return x + self._mean


# TODO: Deprecate in 0.7.0
# These have been maintained for backwards compatibility
LinearModel = LinearVectorModel
MeanLinearModel = MeanLinearVectorModel
