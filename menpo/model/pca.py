import numpy as np
from scipy.linalg.blas import dgemm
from menpo.math import principal_component_decomposition
from menpo.model.base import MeanInstanceLinearModel


class PCAModel(MeanInstanceLinearModel):
    """A :map:`MeanInstanceLinearModel` where components Principal Components.


    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`principal_component_decomposition`.

    Parameters
    ----------
    samples : list of :map:`Vectorizable`
        List of samples to build the model from.
    center : bool, optional
        When True (True by default) PCA is performed after mean centering the
        data. If False the data is assumed to be centred, and the mean will
        be 0.
    bias: bool, optional
        When True (False by default) a biased estimator of the covariance
        matrix is used. See notes.

    ..notes:

    True bias mean that we calculate the covariance as

    :math:`\frac{1}{N} \sum_i^N \mathbf{x}_i \mathbf{x}_i^T`

    instead of default

    :math:`\frac{1}{N-1} \sum_i^N \mathbf{x}_i \mathbf{x}_i^T`

    """
    def __init__(self, samples, center=True, bias=False):
        # build data matrix
        n_samples = len(samples)
        n_features = samples[0].n_parameters
        data = np.zeros((n_samples, n_features))
        for i, sample in enumerate(samples):
            data[i] = sample.as_vector()

        # compute pca
        eigenvectors, eigenvalues, mean_vector = \
            principal_component_decomposition(data, whiten=False,
                                              center=center, bias=bias)

        super(PCAModel, self).__init__(eigenvectors, mean_vector, samples[0])
        self.centered = center
        self.biased = bias
        self._eigenvalues = eigenvalues
        self._n_components = self.n_components
        self._trimmed_eigenvalues = None

    @property
    def n_active_components(self):
        r"""
        The number of components currently in use on this model.
        """
        return self._n_components

    @n_active_components.setter
    def n_active_components(self, value):
        err_str = ("Tried setting n_active_components to {} - "
                   "value needs to be a float "
                   "0.0 < n_components < self._total_kept_variance_ratio "
                   "({}) or an integer 1 < n_components < "
                   "self.n_components ({})".format(
                   value, self._total_variance_ratio, self.n_components))

        # check value
        if isinstance(value, float):
            if 0.0 < value <= self._total_variance_ratio:
                # value needed to capture desired variance
                value = np.sum(
                    [r < value
                     for r in self._total_eigenvalues_cumulative_ratio]) + 1
            else:
                # variance must be bigger than 0.0
                raise ValueError(err_str)
        if isinstance(value, int):
            if value < 1:
                # at least 1 value must be kept
                raise ValueError(err_str)
            elif value >= self.n_components:
                if self.n_active_components < self.n_components:
                    # if the number of available components is smaller than
                    # the total number of components set value to the later
                    value = self.n_components
                else:
                    # if the previous is false and value bigger than the
                    # total number of components, do nothing
                    return

        if 0 < value <= self.n_components:
            self._n_components = value
        else:
            raise ValueError(err_str)

    @MeanInstanceLinearModel.components.getter
    def components(self):
        r"""
        Returns the active components of the model.

        type: (n_active_components, n_features) ndarray
        """
        return self._components[:self.n_active_components, :]

    @property
    def whitened_components(self):
        r"""
        Returns the active components of the model whitened.

        type: (n_active_components, n_features) ndarray
        """
        return self.components / (
            np.sqrt(self.eigenvalues + self.noise_variance)[:, None])

    @property
    def original_variance(self):
        r"""
        Returns the total amount of variance captured by the original model,
        i.e. the amount of variance present on the original samples.

        type: float
        """
        original_variance = self._eigenvalues.sum()
        if self._trimmed_eigenvalues is not None:
            original_variance += self._trimmed_eigenvalues.sum()
        return original_variance

    @property
    def variance(self):
        r"""
        Returns the total amount of variance retained by the active
        components.

        type: float
        """
        return self.eigenvalues.sum()

    @property
    def _total_variance(self):
        r"""
        Returns the total amount of variance retained by all components
        (active and inactive). Useful when the model has been trimmed.

        type: float
        """
        return self._eigenvalues.sum()

    @property
    def variance_ratio(self):
        r"""
        Returns the ratio between the amount of variance retained by the
        active components and the total amount of variance present on the
        original samples.

        type: float
        """
        return self.variance / self.original_variance

    @property
    def _total_variance_ratio(self):
        r"""
        Returns the ratio between the total amount of variance retained by
        all components (active and inactive) and the total amount of variance
        present on the original samples.

        type: float
        """
        return self._total_variance / self.original_variance

    @property
    def eigenvalues(self):
        r"""
        Returns the eigenvalues associated to the active components of the
        model, i.e. the amount of variance captured by each active component.

        type: (n_active_components,) ndarray
        """
        return self._eigenvalues[:self.n_active_components]

    @property
    def eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        type: (n_active_components,) ndarray
        """
        return self.eigenvalues / self.original_variance

    @property
    def _total_eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        type: (n_active_components,) ndarray
        """
        return self._eigenvalues / self.original_variance

    @property
    def eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        type: (n_active_components,) ndarray
        """
        cumulative_ratio = []
        previous_ratio = 0
        for ratio in self.eigenvalues_ratio:
            new_ratio = previous_ratio + ratio
            cumulative_ratio.append(new_ratio)
            previous_ratio = new_ratio
        return cumulative_ratio

    @property
    def _total_eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        type: (n_active_components,) ndarray
        """
        total_cumulative_ratio = []
        previous_ratio = 0
        for ratio in self._total_eigenvalues_ratio:
            new_ratio = previous_ratio + ratio
            total_cumulative_ratio.append(new_ratio)
            previous_ratio = new_ratio
        return total_cumulative_ratio

    @property
    def noise_variance(self):
        r"""
        Returns the average variance captured by the inactive components,
        i.e. the sample noise assumed in a PPCA formulation.

        If all components are active, noise variance is equal to 0.0

        type: float
        """
        if self.n_active_components == self.n_components:
            noise_variance = 0.0
            if self._trimmed_eigenvalues is not None:
                noise_variance += self._trimmed_eigenvalues.mean()
        else:
            if self._trimmed_eigenvalues is not None:
                noise_variance = np.hstack(
                    (self._eigenvalues[self.n_active_components:],
                     self._trimmed_eigenvalues)).mean()
            else:
                noise_variance = (
                    self._eigenvalues[self.n_active_components:].mean())
        return noise_variance

    @property
    def noise_variance_ratio(self):
        r"""
        Returns the ratio between the noise variance and the total amount of
        variance present on the original samples.

        type: float
        """
        return self.noise_variance / self.original_variance

    @property
    def inverse_noise_variance(self):
        r"""
        Returns the inverse of the noise variance.

        type: float
        """
        noise_variance = self.noise_variance
        if noise_variance == 0:
            raise ValueError("noise variance is nil - cannot take the "
                             "inverse")
        return 1.0 / noise_variance

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

            `weights[i, j]` is the linear contribution of the j'th
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
        Permanently trims the components down to a certain amount. The
        number of active components will be automatically reset to this
        particular value.

        This will reduce `self.n_components` down to `n_components`
        (if None `self.n_active_components` will be used), freeing
        up memory in the process.

        Once the model is trimmed, the trimmed components cannot be recovered.

        Parameters
        ----------

        n_components: int >= 1 or float > 0.0, optional
            The number of components that are kept or else the amount (ratio)
            of variance that is kept. If None, `self.n_active_components` is
            used.

        Notes
        -----
        In case `n_components` is greater than the total number of
        components or greater than the amount of variance
        currently kept, this method does not perform any action.
        """
        if n_components is None:
            # by default trim using the current n_active_components
            n_components = self.n_active_components
        # set self.n_active_components to n_components
        self.n_active_components = n_components

        if self.n_active_components < self.n_components:
            # set self.n_components to n_components
            self._components = self._components[:self.n_active_components]
            # store the eigenvalues associated to the discarded components
            self._trimmed_eigenvalues = \
                self._eigenvalues[self.n_active_components:]
            # make sure that the eigenvalues are trimmed too
            self._eigenvalues = self._eigenvalues[:self.n_active_components]

    def distance_to_subspace(self, instance):
        """
        Returns a version of `instance` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the `noise_variance`

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        scaled_projected_out : `self.instance_class`
            A copy of `instance`, with all basis of the model projected out
            and scaled by the inverse of the `noise_variance`.
        """
        vec_instance = self.distance_to_subspace_vector(instance.as_vector())
        return instance.from_vector(vec_instance)

    def distance_to_subspace_vector(self, vector_instance):
        """
        Returns a version of `instance` where all the basis of the model
        have been projected out and which has been scaled by the inverse of
        the `noise_variance`.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        scaled_projected_out: (n_features,) ndarray
            A copy of `vector_instance` with all basis of the model projected
            out and scaled by the inverse of the `noise_variance`.
        """
        return (self.inverse_noise_variance *
                self.project_out_vectors(vector_instance))

    def project_whitened(self, instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of `instance`.

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        sheared_reconstruction : `self.instance_class`
            A sheared (non-orthogonal) reconstruction of `instance`.
        """
        vector_instance = self.project_whitened_vector(instance.as_vector())
        return instance.from_vector(vector_instance)

    def project_whitened_vector(self, vector_instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of
        `vector_instance`.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        sheared_reconstruction : (n_features,) ndarray
            A sheared (non-orthogonal) reconstruction of `vector_instance`
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
            if self.n_active_components < n_available_components:
                # save the current number of active components
                n_active_components = self.n_active_components
            else:
                # save the current number of available components
                n_active_components = n_available_components
            # call trim_components to update our state.
            self.trim_components(n_components=n_available_components)
            if n_active_components < n_available_components:
                # reset the number of active components
                self.n_active_components = n_active_components

        # now we can set our own components with the updated orthogonal ones
        self.components = Q[linear_model.n_components:, :]

    def __str__(self):
        str_out = 'PCA Model \n'
        str_out = str_out + \
            ' - centered:             {}\n' \
            ' - biased:               {}\n' \
            ' - # features:           {}\n' \
            ' - # active components:  {}\n'.format(
            self.centered, self.biased, self.n_features,
            self.n_active_components)
        str_out = str_out + \
            ' - kept variance:        {:.2}  {:.1%}\n' \
            ' - noise variance:       {:.2}  {:.1%}\n'.format(
            self.variance, self.variance_ratio,
            self.noise_variance, self.noise_variance_ratio)
        str_out = str_out + \
            ' - total # components:   {}\n' \
            ' - components shape:     {}\n'.format(
            self.n_components, self.components.shape)
        return str_out
