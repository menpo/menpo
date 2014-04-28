import numpy as np
from scipy.linalg.blas import dgemm
from menpo.decomposition import principal_component_decomposition
from menpo.model.base import MeanInstanceLinearModel


class PCAModel(MeanInstanceLinearModel):
    """
    A :class:`MeanLinearInstanceModel` where the components are Principal
    Components.

    Principal Component Analysis (PCA) by Eigenvalue Decomposition of the
    data's scatter matrix.

    For details of the implementation of PCA, see :func:`menpo.decomposition
    .principal_component_decomposition`.

    Parameters
    ----------
    samples: list of :class:`menpo.base.Vectorizable`
        List of samples to build the model from.

    center : bool, optional
        When True (True by default) PCA is performed after mean centering the
        data. If False the data is assumed to be centred, and the mean will
        be 0.

    bias: bool, optional
        When True (False by default) a biased estimator of the covariance
        matrix is used, i.e.:

            \frac{1}{N} \sum_i^N \mathbf{x}_i \mathbf{x}_i^T

        instead of default:

            \frac{1}{N-1} \sum_i^N \mathbf{x}_i \mathbf{x}_i^T
    """
    def __init__(self, samples, center=True, bias=False):
        self.samples = samples
        self.center = center
        self.bias = bias
        # build data matrix
        n_samples = len(samples)
        n_features = samples[0].n_parameters
        data = np.zeros((n_samples, n_features))
        for i, sample in enumerate(samples):
            data[i] = sample.as_vector()

        eigenvectors, eigenvalues, mean_vector = \
            principal_component_decomposition(data, whiten=False,
                                              center=center, bias=bias)
        self._eigenvalues = eigenvalues

        super(PCAModel, self).__init__(eigenvectors, mean_vector, samples[0])
        self._n_components = self.n_components
        self._trimmed_variance = 0

    @property
    def n_active_components(self):
        r"""
        The number of components currently in use on this model.
        """
        return self._n_components

    @n_active_components.setter
    def n_active_components(self, value):
        value = round(value)
        if 0 < value <= self.n_components:
            self._n_components = value
        else:
            raise ValueError(
                "Tried setting n_components as {} - has to be an int and "
                "0 < n_components <= n_available_components "
                "(which is {}) ".format(value, self.n_components))

    @MeanInstanceLinearModel.components.getter
    def components(self):
        r"""
        The matrix containing the active components on this model.

        type: (n_active_components, n_features) ndarray
        """
        return self._components[:self.n_active_components, :]

    @property
    def whitened_components(self):
        return self.components / (
            np.sqrt(self.eigenvalues + self.noise_variance)[:, None])

    @property
    def n_samples(self):
        return len(self.samples)

    @property
    def eigenvalues(self):
        return self._eigenvalues[:self.n_active_components]

    @property
    def eigenvalues_ratio(self):
        return self.eigenvalues / self.eigenvalues.sum()

    @property
    def noise_variance(self):
        if self.n_active_components == self.n_components:
            return self._trimmed_variance
        else:
            return (self._eigenvalues[self.n_active_components:].mean() +
                    self._trimmed_variance)

    @property
    def inverse_noise_variance(self):
        noise_variance = self.noise_variance
        if noise_variance == 0:
            raise ValueError("noise variance is nil - cannot take the "
                             "inverse")
        else:
            return 1.0 / noise_variance

    @property
    def jacobian(self):
        """
        Returns the Jacobian of the PCA model reshaped to have the standard
        Jacobian shape:

            n_points    x  n_params      x  n_dims
            n_features  x  n_components  x  n_dims

        Returns
        -------
        jacobian : (n_features, n_components, n_dims) ndarray
            The Jacobian of the model in the standard Jacobian shape.
        """
        jacobian = self._jacobian.reshape(self.n_active_components, -1,
                                          self.template_instance.n_dims)
        return jacobian.swapaxes(0, 1)

    @property
    def _jacobian(self):
        """
        Returns the Jacobian of the PCA model with respect to the weights.

        Returns
        -------
        jacobian : (n_components, n_features) ndarray
            The Jacobian with respect to the weights
        """
        return self.components

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
            A scale factor that should be applied to the component. Only
            valid in the case where with_mean is True. The scale is applied
            in units of standard deviations (so a scale of 1.0
            with_mean visualizes the mean plus 1 std. dev of the component
            in question).

        :type: (n_features,) ndarray
        """
        if with_mean:
            # on PCA, scale is in units of std. deviations...
            scaled_eigval = scale * np.sqrt(self.eigenvalues[index])
            return (scaled_eigval * self.components[index]) + self.mean_vector
        else:
            return self.components[index]

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
            principal component to the i'th instance vector produced. Note
            that if n_weights < n_components, only the first n_weight
            components are used in the reconstruction (i.e. unspecified
            weights are implicitly 0)

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
        if n_weights > self.n_active_components:
            raise ValueError(
                "Number of weightings cannot be greater than {}".format(
                    self.n_active_components))
        else:
            full_weights = np.zeros((n_instances, self.n_active_components))
            full_weights[..., :n_weights] = weights
            weights = full_weights
        return self._instance_vectors_for_full_weights(weights)

    def trim_components(self, n_components=None):
        r"""
        Permanently trims the components down to a certain amount.

        This will reduce `n_available_components` down to `n_components`
        (either provided or as currently set), freeing up memory in the
        process.

        Once the model is trimmed, the trimmed components cannot be recovered.

        Parameters
        ----------

        n_components: int, optional
            The number of components that are kept. If None,
            self.n_components is used.
        """
        if n_components is None:
            # by default trim using self.n_components
            n_components = self.n_active_components

        # trim components
        if not n_components < self.n_components:
            raise ValueError(
                "n_components ({}) needs to be less than "
                "n_available_components ({})".format(
                n_components, self.n_components))
        else:
            self._components = self._components[:n_components]
        if self.n_active_components > self.n_components:
            # set n_components if necessary
            self.n_active_components = self.n_components
        # store the amount of variance captured by the discarded components
        self._trimmed_variance = \
            self._eigenvalues[self.n_active_components:].mean()
        # make sure that the eigenvalues are trimmed too
        self._eigenvalues = self._eigenvalues[:self.n_components]

    def distance_to_subspace(self, instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the ``noise_variance``

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        scaled_projected_out : ``self.instance_class``
            A copy of ``instance``, with all basis of the model projected out
            and scaled by the inverse of the ``noise_variance``.
        """
        vec_instance = self.distance_to_subspace_vector(instance.as_vector())
        return instance.from_vector(vec_instance)

    def distance_to_subspace_vector(self, vector_instance):
        """
        Returns a version of ``instance`` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the ``noise_variance``.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        scaled_projected_out: (n_features,) ndarray
            A copy of ``vector_instance`` with all basis of the model projected
            out and scaled by the inverse of the ``noise_variance``.
        """
        return (self.inverse_noise_variance *
                self.project_out_vectors(vector_instance))

    def project_whitened(self, instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of ``instance``.

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        sheared_reconstruction : ``self.instance_class``
            A sheared (non-orthogonal) reconstruction of ``instance``.
        """
        vector_instance = self.project_whitened_vector(instance.as_vector())
        return instance.from_vector(vector_instance)

    def project_whitened_vector(self, vector_instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of
        ``vector_instance``.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        sheared_reconstruction : (n_features,) ndarray
            A sheared (non-orthogonal) reconstruction of ``vector_instance``
        """
        whitened_components = self.whitened_components
        weights = dgemm(alpha=1.0, a=vector_instance.T,
                        b=whitened_components.T, trans_a=True)
        return dgemm(alpha=1.0, a=weights.T, b=whitened_components.T,
                     trans_a=True, trans_b=True)

    def orthonormalize_against_inplace(self, linear_model):
        r"""
        Enforces that the union of this model's components and another are
        both mutually orthonormal.

        Note that the model passed in is guaranteed to not have it's number
        of available components changed. This model, however, may loose some
        dimensionality due to reaching a degenerate state.

        The removed components will always be trimmed from the end of
        components (i.e. the components which capture the least variance).
        If trimming is performed, `n_components` and
        `n_available_components` would be altered - see
        :meth:`trim_components` for details.

        Parameters
        -----------
        linear_model : :class:`LinearModel`
            A second linear model to orthonormalize this against.
        """
        # take the QR decomposition of the model components
        Q = (np.linalg.qr(np.hstack((linear_model._components.T,
                                     self._components.T)))[0]).T
        # the model passed to us went first, so all it's components will
        # survive. Pull them off, and update the other model.
        linear_model.components = Q[:linear_model.n_components, :]
        # it's possible that all of our components didn't survive due to
        # degeneracy. We need to trim our components down before replacing
        # them to ensure the number of components is consistent (otherwise
        # the components setter will complain at us)
        n_available_components = Q.shape[0] - linear_model.n_components
        if n_available_components < self.n_components:
            # oh dear, we've lost some components from the end of our model.
            # call trim_components to update our state.
            self.trim_components(n_components=n_available_components)
        # now we can set our own components with the updated orthogonal ones
        self.components = Q[linear_model.n_components:, :]
