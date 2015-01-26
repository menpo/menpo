from __future__ import division
import numpy as np
from menpo.math import pca, ipca
from menpo.model.base import MeanInstanceLinearModel
from menpo.visualize import print_dynamic, progress_bar_str


class PCAModel(MeanInstanceLinearModel):
    """A :map:`MeanInstanceLinearModel` where components are Principal
    Components.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`pca`.

    Parameters
    ----------
    samples : list of :map:`Vectorizable`
        List of samples to build the model from.
    centre : bool, optional
        When True (True by default) PCA is performed after mean centering the
        data. If False the data is assumed to be centred, and the mean will
        be 0.
    n_samples : int, optional
        If provided then ``samples``  must be an iterator  that yields
        ``n_samples``. If not provided then samples has to be a
        list (so we know how large the data matrix needs to be).
    """
    def __init__(self, samples, centre=True, n_samples=None, verbose=False):
        # extract data matrix, template and number of samples
        data, template, self.n_samples = extract_data(
            samples, n_samples=n_samples, verbose=verbose)

        # compute pca
        e_vectors, e_values, mean = pca(data, centre=centre, inplace=True)

        super(PCAModel, self).__init__(e_vectors, mean, template)
        self.centred = centre
        self._eigenvalues = e_values
        # start the active components as all the components
        self._n_active_components = int(self.n_components)
        self._trimmed_eigenvalues = np.array([])

    @property
    def n_active_components(self):
        r"""
        The number of components currently in use on this model.
        """
        return self._n_active_components

    @n_active_components.setter
    def n_active_components(self, value):
        err_str = ("Tried setting n_active_components to {} - "
                   "value needs to be a float "
                   "0.0 < n_components < self._total_kept_variance_ratio "
                   "({}) or an integer 1 < n_components < "
                   "self.n_components ({})".format(
                   value, self._total_variance_ratio(), self.n_components))

        # check value
        if isinstance(value, float):
            if 0.0 < value <= self._total_variance_ratio():
                # value needed to capture desired variance
                value = np.sum(
                    [r < value
                     for r in self._total_eigenvalues_cumulative_ratio()]) + 1
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
            self._n_active_components = int(value)
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
    def eigenvalues(self):
        r"""
        Returns the eigenvalues associated to the active components of the
        model, i.e. the amount of variance captured by each active component.

        type: (n_active_components,) ndarray
        """
        return self._eigenvalues[:self.n_active_components]

    def whitened_components(self):
        r"""
        Returns the active components of the model whitened.

        type: (n_active_components, n_features) ndarray
        """
        return self.components / (
            np.sqrt(self.eigenvalues * self.n_samples +
                    self.noise_variance())[:, None])

    def original_variance(self):
        r"""
        Returns the total amount of variance captured by the original model,
        i.e. the amount of variance present on the original samples.

        type: float
        """
        return self._eigenvalues.sum() + self._trimmed_eigenvalues.sum()

    def variance(self):
        r"""
        Returns the total amount of variance retained by the active
        components.

        type: float
        """
        return self.eigenvalues.sum()

    def _total_variance(self):
        r"""
        Returns the total amount of variance retained by all components
        (active and inactive). Useful when the model has been trimmed.

        type: float
        """
        return self._eigenvalues.sum()

    def variance_ratio(self):
        r"""
        Returns the ratio between the amount of variance retained by the
        active components and the total amount of variance present on the
        original samples.

        type: float
        """
        return self.variance() / self.original_variance()

    def _total_variance_ratio(self):
        r"""
        Returns the ratio between the total amount of variance retained by
        all components (active and inactive) and the total amount of variance
        present on the original samples.

        type: float
        """
        return self._total_variance() / self.original_variance()

    def eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        type: (n_active_components,) ndarray
        """
        return self.eigenvalues / self.original_variance()

    def _total_eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        type: (n_active_components,) ndarray
        """
        return self._eigenvalues / self.original_variance()

    def eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        type: (n_active_components,) ndarray
        """
        cumulative_ratio = []
        previous_ratio = 0
        for ratio in self.eigenvalues_ratio():
            new_ratio = previous_ratio + ratio
            cumulative_ratio.append(new_ratio)
            previous_ratio = new_ratio
        return cumulative_ratio

    def _total_eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        type: (n_active_components,) ndarray
        """
        total_cumulative_ratio = []
        previous_ratio = 0
        for ratio in self._total_eigenvalues_ratio():
            new_ratio = previous_ratio + ratio
            total_cumulative_ratio.append(new_ratio)
            previous_ratio = new_ratio
        return total_cumulative_ratio

    def noise_variance(self):
        r"""
        Returns the average variance captured by the inactive components,
        i.e. the sample noise assumed in a PPCA formulation.

        If all components are active, noise variance is equal to 0.0

        type: float
        """
        if self.n_active_components == self.n_components:
            noise_variance = 0.0
            if self._trimmed_eigenvalues.size is not 0:
                noise_variance += self._trimmed_eigenvalues.mean()
        else:
            noise_variance = np.hstack(
                (self._eigenvalues[self.n_active_components:],
                 self._trimmed_eigenvalues)).mean()
        return noise_variance

    def noise_variance_ratio(self):
        r"""
        Returns the ratio between the noise variance and the total amount of
        variance present on the original samples.

        type: float
        """
        return self.noise_variance() / self.original_variance()

    def inverse_noise_variance(self):
        r"""
        Returns the inverse of the noise variance.

        type: float
        """
        noise_variance = self.noise_variance()
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
            self._trimmed_eigenvalues = np.hstack((
                self._trimmed_eigenvalues,
                self._eigenvalues[self.n_active_components:]))
            # make sure that the eigenvalues are trimmed too
            self._eigenvalues = self._eigenvalues[:self.n_active_components]

    def project_whitened(self, instance):
        """
        Projects the `instance` onto the whitened components, retrieving the 
        whitened linear weightings.

        Parameters
        ----------
        instance : :class:`menpo.base.Vectorizable`
            A novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of whitened linear weightings
        """
        return self.project_whitened_vector(instance.as_vector())

    def project_whitened_vector(self, vector_instance):
        """
        Projects the `vector_instance` onto the whitened components, 
        retrieving the whitened linear weightings.

        Parameters
        ----------
        vector_instance : (n_features,) ndarray
            A novel vector.

        Returns
        -------
        projected : (n_components,)
            A vector of whitened linear weightings
        """
        whitened_components = self.whitened_components()
        return np.dot(vector_instance, whitened_components.T)

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

    def increment(self, samples, n_samples=None, forgetting_factor=1.0,
                  verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        -----------
        samples : list of :map:`Vectorizable`
            List of new samples to update the model from.
        n_samples : int, optional
            If provided then ``samples``  must be an iterator  that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        forgetting_factor : [0.0, 1.0] float, optional
            Forgetting factor that weights the relative contribution of new
            samples vs old samples. If 1.0, all samples are weighted equally
            and, hence, the results is the exact same as performing batch
            PCA on the concatenated list of old and new simples. If <1.0,
            more emphasis is put on the new samples. See [1] for details.

        References
        ----------
        .. [1] David Ross, Jongwoo Lim, Ruei-Sung Lin, Ming-Hsuan Yang.
           "Incremental Learning for Robust Visual Tracking". IJCV, 2007.
        """
        # extract data matrix, template and number of samples
        data, template, n_samples = extract_data(
            samples, n_samples=n_samples, verbose=verbose)

        # compute incremental pca
        e_vectors, e_values, m_vector = ipca(
            data, self._components, self._eigenvalues, self.n_samples,
            m_a=self.mean_vector, f=forgetting_factor)

        # if the number of active components is the same as the total number
        # of components so it will be after this method is executed
        reset = 1 if self.n_active_components == self.n_components else 0

        # update mean, components, eigenvalues and number of samples
        self.mean_vector = m_vector
        self._components = e_vectors
        self._eigenvalues = e_values
        self.n_samples += n_samples

        # reset the number of active components to the total number of
        # components
        if reset:
            self.n_active_components = self.n_components

    def __str__(self):
        str_out = 'PCA Model \n'
        str_out = str_out + \
            ' - centred:             {}\n' \
            ' - # features:           {}\n' \
            ' - # active components:  {}\n'.format(
            self.centred, self.n_features,
            self.n_active_components)
        str_out = str_out + \
            ' - kept variance:        {:.2}  {:.1%}\n' \
            ' - noise variance:       {:.2}  {:.1%}\n'.format(
            self.variance(), self.variance_ratio(),
            self.noise_variance(), self.noise_variance_ratio())
        str_out = str_out + \
            ' - total # components:   {}\n' \
            ' - components shape:     {}\n'.format(
            self.n_components, self.components.shape)
        return str_out


def extract_data(samples, n_samples=None, verbose=False):
    # get the first element as the template and use it to configure the
    # data matrix
    if n_samples is None:
        # samples is a list
        n_samples = len(samples)
        template = samples[0]
        samples = samples[1:]
    else:
        # samples is an iterator
        template = next(samples)
    n_features = template.n_parameters
    template_vector = template.as_vector()
    data = np.zeros((n_samples, n_features), dtype=template_vector.dtype)
    # now we can fill in the first element from the template
    data[0] = template_vector
    del template_vector
    if verbose:
        print('Allocated data matrix {:.2f}'
              'GB'.format(data.nbytes / 2 ** 30))
    # 1-based as we have the template vector set already
    for i, sample in enumerate(samples, 1):
        if i >= n_samples:
            break
        if verbose:
            print_dynamic(
                'Building data matrix from {} samples - {}'.format(
                    n_samples,
                progress_bar_str(float(i + 1) / n_samples, show_bar=True)))
        data[i] = sample.as_vector()

    return data, template, n_samples
