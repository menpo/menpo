from __future__ import division
import numpy as np

from menpo.base import doc_inherit
from menpo.math import pca, pcacov, ipca, as_matrix
from .linear import MeanLinearVectorModel
from .vectorizable import VectorizableBackedModel


class PCAVectorModel(MeanLinearVectorModel):
    r"""
    A :map:`MeanLinearModel` where components are Principal Components.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`pca`.

    Parameters
    ----------
    samples : `ndarray` or `list` or `iterable` of `ndarray`
        List or iterable of numpy arrays to build the model from, or an
        existing data matrix.
    centre : `bool`, optional
        When ``True`` (default) PCA is performed after mean centering the data.
        If ``False`` the data is assumed to be centred, and the mean will be
        ``0``.
    n_samples : `int`, optional
        If provided then ``samples`` must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
    max_n_components : `int`, optional
        The maximum number of components to keep in the model. Any components
        above and beyond this one are discarded.
    inplace : `bool`, optional
        If ``True`` the data matrix is modified in place. Otherwise, the data
        matrix is copied.
    """
    def __init__(self, samples, centre=True, n_samples=None,
                 max_n_components=None, inplace=True):
        # Generate data matrix
        data, self.n_samples = self._data_to_matrix(samples, n_samples)

        # Compute pca
        e_vectors, e_values, mean = pca(data, centre=centre, inplace=inplace)

        # The call to __init__ of MeanLinearModel is done in here
        self._constructor_helper(
            eigenvalues=e_values, eigenvectors=e_vectors, mean=mean,
            centred=centre, max_n_components=max_n_components)

    @classmethod
    def init_from_covariance_matrix(cls, C, mean, n_samples, centred=True,
                                    is_inverse=False, max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) by eigenvalue
        decomposition of the provided covariance/scatter matrix. For details
        of the implementation of PCA, see :map:`pcacov`.

        Parameters
        ----------
        C : ``(n_features, n_features)`` `ndarray` or `scipy.sparse`
            The Covariance/Scatter matrix. If it is a precision matrix (inverse
            covariance), then set `is_inverse=True`.
        mean : ``(n_features, )`` `ndarray`
            The mean vector.
        n_samples : `int`
            The number of samples used to generate the covariance matrix.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the covariance matrix.
        is_inverse : `bool`, optional
            It ``True``, then it is assumed that `C` is a precision matrix (
            inverse covariance). Thus, the eigenvalues will be inverted. If
            ``False``, then it is assumed that `C` is a covariance matrix.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # Compute pca on covariance
        e_vectors, e_values = pcacov(C, is_inverse=is_inverse)

        # Create new pca instance
        model = PCAModel.__new__(cls)
        model.n_samples = n_samples

        # The call to __init__ of MeanLinearModel is done in here
        model._constructor_helper(
            eigenvalues=e_values, eigenvectors=e_vectors, mean=mean,
            centred=centred, max_n_components=max_n_components)
        return model

    @classmethod
    def init_from_components(cls, components, eigenvalues, mean, n_samples,
                             centred, max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) using the provided
        components (eigenvectors) and eigenvalues.

        Parameters
        ----------
        components : ``(n_components, n_features)`` `ndarray`
            The eigenvectors to be used.
        eigenvalues : ``(n_components, )`` `ndarray`
            The corresponding eigenvalues.
        mean : ``(n_features, )`` `ndarray`
            The mean vector.
        n_samples : `int`
            The number of samples used to generate the eigenvectors.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the eigenvectors.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # This is a bit of a filthy trick that by rights should not be done,
        # but we want to have these nice static constructors so we are living
        # with the shame (create an empty object instance which we fill in).
        model = PCAModel.__new__(cls)
        model.n_samples = n_samples

        # The call to __init__ of MeanLinearModel is done in here
        model._constructor_helper(
            eigenvalues=eigenvalues, eigenvectors=components, mean=mean,
            centred=centred, max_n_components=max_n_components)
        return model

    def _constructor_helper(self, eigenvalues, eigenvectors, mean, centred,
                            max_n_components):
        # if covariance is not centred, mean must be zeros.
        if centred:
            MeanLinearVectorModel.__init__(self, eigenvectors, mean)
        else:
            MeanLinearVectorModel.__init__(self, eigenvectors,
                                           np.zeros(mean.shape, dtype=mean.dtype))
        self.centred = centred
        self._eigenvalues = eigenvalues
        # start the active components as all the components
        self._n_active_components = int(self.n_components)
        self._trimmed_eigenvalues = np.array([])
        if max_n_components is not None:
            self.trim_components(max_n_components)

    def _data_to_matrix(self, data, n_samples):
        # build a data matrix from all the samples
        if n_samples is None:
            n_samples = len(data)
        # Assumed data is ndarray of (n_samples, n_features) or list of samples
        if not isinstance(data, np.ndarray):
            # Make sure we have an array, slice of the number of requested
            # samples
            data = np.array(data)[:n_samples]
        return data, n_samples

    @property
    def n_active_components(self):
        r"""
        The number of components currently in use on this model.

        :type: `int`
        """
        return self._n_active_components

    @n_active_components.setter
    def n_active_components(self, value):
        r"""
        Sets an updated number of active components on this model. The number
        of active components represents the number of principal components
        that will be used for generative purposes. Note that this therefore
        makes the model stateful. Also note that setting the number of
        components will not affect memory unless :meth:`trim_components`
        is called.

        Parameters
        ----------
        value : `int`
            The new number of active components.

        Raises
        ------
        ValueError
            Tried setting n_active_components to {value} - value needs to be a
            float 0.0 < n_components < self._total_kept_variance_ratio ({}) or
            an integer 1 < n_components < self.n_components ({})
        """
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

    @MeanLinearVectorModel.components.getter
    def components(self):
        r"""
        Returns the active components of the model.

        :type: ``(n_active_components, n_features)`` `ndarray`
        """
        return self._components[:self.n_active_components, :]

    @property
    def eigenvalues(self):
        r"""
        Returns the eigenvalues associated with the active components of the
        model, i.e. the amount of variance captured by each active component,
        sorted form largest to smallest.

        :type: ``(n_active_components,)`` `ndarray`
        """
        return self._eigenvalues[:self.n_active_components]

    def whitened_components(self):
        r"""
        Returns the active components of the model, whitened.

        Returns
        -------
        whitened_components : ``(n_active_components, n_features)`` `ndarray`
            The whitened components.
        """
        return self.components / (
            np.sqrt(self.eigenvalues * self.n_samples +
                    self.noise_variance())[:, None])

    def original_variance(self):
        r"""
        Returns the total amount of variance captured by the original model,
        i.e. the amount of variance present on the original samples.

        Returns
        -------
        optional_variance : `float`
            The variance captured by the model.
        """
        return self._eigenvalues.sum() + self._trimmed_eigenvalues.sum()

    def variance(self):
        r"""
        Returns the total amount of variance retained by the active
        components.

        Returns
        -------
        variance : `float`
            Total variance captured by the active components.
        """
        return self.eigenvalues.sum()

    def _total_variance(self):
        r"""
        Returns the total amount of variance retained by all components
        (active and inactive). Useful when the model has been trimmed.

        Returns
        -------
        total_variance : `float`
            Total variance captured by all components.
        """
        return self._eigenvalues.sum()

    def variance_ratio(self):
        r"""
        Returns the ratio between the amount of variance retained by the
        active components and the total amount of variance present on the
        original samples.

        Returns
        -------
        variance_ratio : `float`
            Ratio of active components variance and total variance present
            in original samples.
        """
        return self.variance() / self.original_variance()

    def _total_variance_ratio(self):
        r"""
        Returns the ratio between the total amount of variance retained by
        all components (active and inactive) and the total amount of variance
        present on the original samples.

        Returns
        -------
        total_variance_ratio : `float`
            Ratio of total variance over the original variance.
        """
        return self._total_variance() / self.original_variance()

    def eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        Returns
        -------
        eigenvalues_ratio : ``(n_active_components,)`` `ndarray`
            The active eigenvalues array scaled by the original variance.
        """
        return self.eigenvalues / self.original_variance()

    def _total_eigenvalues_ratio(self):
        r"""
        Returns the ratio between the variance captured by each active
        component and the total amount of variance present on the original
        samples.

        Returns
        -------
        total_eigenvalues_ratio : ``(n_components,)`` `ndarray`
            Array of eigenvalues scaled by the original variance.
        """
        return self._eigenvalues / self.original_variance()

    def eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        Returns
        -------
        eigenvalues_cumulative_ratio : ``(n_active_components,)`` `ndarray`
            Array of cumulative eigenvalues.
        """
        return np.cumsum(self.eigenvalues_ratio())

    def _total_eigenvalues_cumulative_ratio(self):
        r"""
        Returns the cumulative ratio between the variance captured by the
        active components and the total amount of variance present on the
        original samples.

        Returns
        -------
        total_eigenvalues_cumulative_ratio : ``(n_active_components,)`` `ndarray`
            Array of total cumulative eigenvalues.
        """
        return np.cumsum(self._total_eigenvalues_ratio())

    def noise_variance(self):
        r"""
        Returns the average variance captured by the inactive components,
        i.e. the sample noise assumed in a Probabilistic PCA formulation.

        If all components are active, then ``noise_variance == 0.0``.

        Returns
        -------
        noise_variance : `float`
            The mean variance of the inactive components.
        """
        if self.n_active_components == self.n_components:
            if self._trimmed_eigenvalues.size != 0:
                noise_variance = self._trimmed_eigenvalues.mean()
            else:
                noise_variance = 0.0
        else:
            noise_variance = np.hstack(
                (self._eigenvalues[self.n_active_components:],
                 self._trimmed_eigenvalues)).mean()
        return noise_variance

    def noise_variance_ratio(self):
        r"""
        Returns the ratio between the noise variance and the total amount of
        variance present on the original samples.

        Returns
        -------
        noise_variance_ratio : `float`
            The ratio between the noise variance and the variance present
            in the original samples.
        """
        return self.noise_variance() / self.original_variance()

    def inverse_noise_variance(self):
        r"""
        Returns the inverse of the noise variance.

        Returns
        -------
        inverse_noise_variance : `float`
            Inverse of the noise variance.

        Raises
        ------
        ValueError
            If ``noise_variance() == 0``
        """
        noise_variance = self.noise_variance()
        if np.allclose(noise_variance, 0):
            raise ValueError("noise variance is effectively 0 - "
                             "cannot take the inverse")
        return 1.0 / noise_variance

    def component(self, index, with_mean=True, scale=1.0):
        r"""
        A particular component of the model, in vectorized form.

        Parameters
        ----------
        index : `int`
            The component that is to be returned
        with_mean: `bool`, optional
            If ``True``, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.
        scale : `float`, optional
            A scale factor that should be applied to the component. Only
            valid in the case where with_mean is ``True``. The scale is applied
            in units of standard deviations (so a scale of ``1.0``
            `with_mean` visualizes the mean plus ``1`` std. dev of the component
            in question).

        Returns
        -------
        component_vector : ``(n_features,)`` `ndarray`
            The component vector of the given index.
        """
        if with_mean:
            # on PCA, scale is in units of std. deviations...
            scaled_eigval = scale * np.sqrt(self.eigenvalues[index])
            return (scaled_eigval * self.components[index]) + self._mean
        else:
            return self.components[index]

    def instance_vectors(self, weights, normalized_weights=False):
        """
        Creates new vectorized instances of the model using the first
        components in a particular weighting.

        Parameters
        ----------
        weights : ``(n_vectors, n_weights)`` `ndarray` or `list` of `lists`
            The weightings for the first `n_weights` components that
            should be used per instance that is to be produced

            ``weights[i, j]`` is the linear contribution of the j'th
            principal component to the i'th instance vector produced. Note
            that if ``n_weights < n_components``, only the first ``n_weight``
            components are used in the reconstruction (i.e. unspecified
            weights are implicitly ``0``).
        normalized_weights : `bool`, optional
            If ``True``, the weights are assumed to be normalized w.r.t the
            eigenvalues. This can be easier to create unique instances by
            making the weights more interpretable.

        Returns
        -------
        vectors : ``(n_vectors, n_features)`` `ndarray`
            The instance vectors for the weighting provided.

        Raises
        ------
        ValueError
            If n_weights > n_components
        """
        weights = np.asarray(weights)  # if eg a list is provided
        n_instances, n_weights = weights.shape
        if n_weights > self.n_active_components:
            raise ValueError(
                "Number of weightings cannot be greater than {}".format(
                    self.n_active_components))
        else:
            full_weights = np.zeros((n_instances, self.n_active_components),
                                    dtype=self._components.dtype)
            full_weights[..., :n_weights] = weights
            weights = full_weights

        if normalized_weights:
            # If the weights were normalized, then they are all relative to
            # to the scale of the eigenvalues and thus must be multiplied by
            # the sqrt of the eigenvalues.
            weights *= self.eigenvalues ** 0.5
        return self._instance_vectors_for_full_weights(weights)

    def instance(self, weights, normalized_weights=False):
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
        normalized_weights : `bool`, optional
            If ``True``, the weights are assumed to be normalized w.r.t the
            eigenvalues. This can be easier to create unique instances by
            making the weights more interpretable.

        Returns
        -------
        vector : ``(n_features,)`` `ndarray`
            The instance vector for the weighting provided.
        """
        weights = np.asarray(weights)
        return self.instance_vectors(
            weights[None, :], normalized_weights=normalized_weights).flatten()

    def trim_components(self, n_components=None):
        r"""
        Permanently trims the components down to a certain amount. The number of
        active components will be automatically reset to this particular value.

        This will reduce `self.n_components` down to `n_components`
        (if ``None``, `self.n_active_components` will be used), freeing up
        memory in the process.

        Once the model is trimmed, the trimmed components cannot be recovered.

        Parameters
        ----------
        n_components: `int` >= ``1`` or `float` > ``0.0`` or ``None``, optional
            The number of components that are kept or else the amount (ratio)
            of variance that is kept. If ``None``, `self.n_active_components` is
            used.

        Notes
        -----
        In case `n_components` is greater than the total number of components or
        greater than the amount of variance currently kept, this method does
        not perform any action.
        """
        if n_components is None:
            # by default trim using the current n_active_components
            n_components = self.n_active_components
        # set self.n_active_components to n_components
        self.n_active_components = n_components

        if self.n_active_components < self.n_components:
            # Just stored so that we can fit < 80 chars
            nac = self.n_active_components
            # set self.n_components to n_components. We have to copy to ensure
            # that the data is actually removed, otherwise a view is returned
            self._components = self._components[:nac].copy()
            # store the eigenvalues associated to the discarded components
            self._trimmed_eigenvalues = np.hstack((
                self._trimmed_eigenvalues,
                self._eigenvalues[self.n_active_components:]))
            # make sure that the eigenvalues are trimmed too
            self._eigenvalues = self._eigenvalues[:nac].copy()

    def project_whitened(self, vector_instance):
        """
        Projects the `vector_instance` onto the whitened components,
        retrieving the whitened linear weightings.

        Parameters
        ----------
        vector_instance : ``(n_features,)`` `ndarray`
            A novel vector.

        Returns
        -------
        projected : ``(n_features,)`` `ndarray`
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
        If trimming is performed, `n_components` and `n_available_components`
        would be altered - see :meth:`trim_components` for details.

        Parameters
        ----------
        linear_model : :map:`LinearModel`
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

    def increment(self, data, n_samples=None, forgetting_factor=1.0,
                  verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        samples : `list` of :map:`Vectorizable`
            List of new samples to update the model from.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
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
        data, n_new_samples = self._data_to_matrix(data, n_samples)

        # compute incremental pca
        e_vectors, e_values, m_vector = ipca(
            data, self._components, self._eigenvalues, self.n_samples,
            m_a=self._mean, f=forgetting_factor)

        # if the number of active components is the same as the total number
        # of components so it will be after this method is executed
        reset = (self.n_active_components == self.n_components)

        # update mean, components, eigenvalues and number of samples
        self._mean = m_vector
        self._components = e_vectors
        self._eigenvalues = e_values
        self.n_samples += n_new_samples

        # reset the number of active components to the total number of
        # components
        if reset:
            self.n_active_components = self.n_components

    def plot_eigenvalues(self, figure_id=None, new_figure=False,
                         render_lines=True, line_colour='b', line_style='-',
                         line_width=2, render_markers=True, marker_style='o',
                         marker_size=6, marker_face_colour='b',
                         marker_edge_colour='k', marker_edge_width=1.,
                         render_axes=True, axes_font_name='sans-serif',
                         axes_font_size=10, axes_font_style='normal',
                         axes_font_weight='normal', figure_size=(10, 6),
                         render_grid=True, grid_line_style='--',
                         grid_line_width=0.5):
        r"""
        Plot of the eigenvalues.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers.
            Example options ::

                {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                 ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                 ``1``, ``2``, ``3``, ``4``, ``8``}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                 ``monospace``}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {``ultralight``, ``light``, ``normal``, ``regular``,
                 ``book``, ``medium``, ``roman``, ``semibold``,
                 ``demibold``, ``demi``, ``bold``, ``heavy``,
                 ``extra bold``, ``black``}

        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        viewer : :map:`MatplotlibRenderer`
            The viewer object.
        """
        from menpo.visualize import plot_curve
        return plot_curve(
            range(self.n_active_components), [self.eigenvalues],
            figure_id=figure_id, new_figure=new_figure, legend_entries=None,
            title='Eigenvalues', x_label='Component Number',
            y_label='Eigenvalue',
            axes_x_limits=[0, self.n_active_components - 1],
            axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid, grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)

    def plot_eigenvalues_widget(self, figure_size=(10, 6), style='coloured'):
        r"""
        Plot of the eigenvalues using an interactive widget.

        Parameters
        ----------
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import plot_graph
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()
        plot_graph(x_axis=range(self.n_active_components),
                   y_axis=[self.eigenvalues], legend_entries=['Eigenvalues'],
                   figure_size=figure_size, style=style)

    def plot_eigenvalues_ratio(self, figure_id=None, new_figure=False,
                               render_lines=True, line_colour='b',
                               line_style='-', line_width=2,
                               render_markers=True, marker_style='o',
                               marker_size=6, marker_face_colour='b',
                               marker_edge_colour='k', marker_edge_width=1.,
                               render_axes=True, axes_font_name='sans-serif',
                               axes_font_size=10, axes_font_style='normal',
                               axes_font_weight='normal', figure_size=(10, 6),
                               render_grid=True, grid_line_style='--',
                               grid_line_width=0.5):
        r"""
        Plot of the variance ratio captured by the eigenvalues.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers.
            Example options ::

                {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                 ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                 ``1``, ``2``, ``3``, ``4``, ``8``}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                 ``monospace``}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {``ultralight``, ``light``, ``normal``, ``regular``,
                 ``book``, ``medium``, ``roman``, ``semibold``,
                 ``demibold``, ``demi``, ``bold``, ``heavy``,
                 ``extra bold``, ``black``}

        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        viewer : :map:`MatplotlibRenderer`
            The viewer object.
        """
        from menpo.visualize import plot_curve
        return plot_curve(
            range(self.n_active_components), [self.eigenvalues_ratio()],
            figure_id=figure_id, new_figure=new_figure, legend_entries=None,
            title='Variance Ratio of Eigenvalues', x_label='Component Number',
            y_label='Variance Ratio',
            axes_x_limits=[0, self.n_active_components - 1],
            axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid, grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)

    def plot_eigenvalues_ratio_widget(self, figure_size=(10, 6),
                                      style='coloured'):
        r"""
        Plot of the variance ratio captured by the eigenvalues using an
        interactive widget.

        Parameters
        ----------
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import plot_graph
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()
        plot_graph(x_axis=range(self.n_active_components),
                   y_axis=[self.eigenvalues_ratio()],
                   legend_entries=['Eigenvalues ratio'],
                   figure_size=figure_size, style=style)

    def plot_eigenvalues_cumulative_ratio(self, figure_id=None,
                                          new_figure=False, render_lines=True,
                                          line_colour='b', line_style='-',
                                          line_width=2, render_markers=True,
                                          marker_style='o', marker_size=6,
                                          marker_face_colour='b',
                                          marker_edge_colour='k',
                                          marker_edge_width=1.,
                                          render_axes=True,
                                          axes_font_name='sans-serif',
                                          axes_font_size=10,
                                          axes_font_style='normal',
                                          axes_font_weight='normal',
                                          figure_size=(10, 6), render_grid=True,
                                          grid_line_style='--',
                                          grid_line_width=0.5):
        r"""
        Plot of the cumulative variance ratio captured by the eigenvalues.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_lines : `bool`, optional
            If ``True``, the line will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers.
            Example options ::

                {``.``, ``,``, ``o``, ``v``, ``^``, ``<``, ``>``, ``+``,
                 ``x``, ``D``, ``d``, ``s``, ``p``, ``*``, ``h``, ``H``,
                 ``1``, ``2``, ``3``, ``4``, ``8``}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {``r``, ``g``, ``b``, ``c``, ``m``, ``k``, ``w``}
                or 
                ``(3, )`` `ndarray`
                or
                `list` of length ``3``

        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {``serif``, ``sans-serif``, ``cursive``, ``fantasy``,
                 ``monospace``}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {``ultralight``, ``light``, ``normal``, ``regular``,
                 ``book``, ``medium``, ``roman``, ``semibold``,
                 ``demibold``, ``demi``, ``bold``, ``heavy``,
                 ``extra bold``, ``black``}

        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.
        render_grid : `bool`, optional
            If ``True``, the grid will be rendered.
        grid_line_style : {``-``, ``--``, ``-.``, ``:``}, optional
            The style of the grid lines.
        grid_line_width : `float`, optional
            The width of the grid lines.

        Returns
        -------
        viewer : :map:`MatplotlibRenderer`
            The viewer object.
        """
        from menpo.visualize import plot_curve
        return plot_curve(
            range(self.n_active_components),
            [self.eigenvalues_cumulative_ratio()], figure_id=figure_id,
            new_figure=new_figure, legend_entries=None,
            title='Cumulative Variance Ratio of Eigenvalues',
            x_label='Component Number', y_label='Cumulative Variance Ratio',
            axes_x_limits=[0, self.n_active_components - 1],
            axes_y_limits=None, axes_x_ticks=None, axes_y_ticks=None,
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, figure_size=figure_size,
            render_grid=render_grid, grid_line_style=grid_line_style,
            grid_line_width=grid_line_width)

    def plot_eigenvalues_cumulative_ratio_widget(self, figure_size=(10, 6),
                                                 style='coloured'):
        r"""
        Plot of the cumulative variance ratio captured by the eigenvalues using
        an interactive widget.

        Parameters
        ----------
        figure_size : (`float`, `float`) or ``None``, optional
            The size of the figure in inches.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import plot_graph
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()
        plot_graph(x_axis=range(self.n_active_components),
                   y_axis=[self.eigenvalues_cumulative_ratio()],
                   legend_entries=['Eigenvalues cumulative ratio'],
                   figure_size=figure_size, style=style)

    def __str__(self):
        str_out = 'PCA Vector Model \n'                      \
                  ' - centred:              {}\n'            \
                  ' - # features:           {}\n'            \
                  ' - # active components:  {}\n'            \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n'            \
                  ' - components shape:     {}\n'.format(
            self.centred,  self.n_features, self.n_active_components,
            self.variance(), self.variance_ratio(), self.noise_variance(),
            self.noise_variance_ratio(), self.n_components,
            self.components.shape)
        return str_out


class PCAModel(VectorizableBackedModel, PCAVectorModel):
    r"""
    A :map:`MeanLinearModel` where components are Principal Components
    and the components are vectorized instances.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`pca`.

    Parameters
    ----------
    samples : `list` or `iterable` of :map:`Vectorizable`
        List or iterable of samples to build the model from.
    centre : `bool`, optional
        When ``True`` (default) PCA is performed after mean centering the data.
        If ``False`` the data is assumed to be centred, and the mean will be
        ``0``.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).
    max_n_components : `int`, optional
        The maximum number of components to keep in the model. Any components
        above and beyond this one are discarded.
    inplace : `bool`, optional
        If ``True`` the data matrix is modified in place. Otherwise, the data
        matrix is copied.
    verbose : `bool`, optional
        Whether to print building information or not.
     """

    def __init__(self, samples, centre=True, n_samples=None,
                 max_n_components=None, inplace=True, verbose=False):
        # build a data matrix from all the samples
        data, template = as_matrix(samples, length=n_samples,
                                   return_template=True, verbose=verbose)
        n_samples = data.shape[0]

        PCAVectorModel.__init__(self, data, centre=centre,
                                max_n_components=max_n_components,
                                n_samples=n_samples, inplace=inplace)
        VectorizableBackedModel.__init__(self, template)

    @classmethod
    def init_from_covariance_matrix(cls, C, mean, n_samples, centred=True,
                                    is_inverse=False, max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) by eigenvalue
        decomposition of the provided covariance/scatter matrix. For details
        of the implementation of PCA, see :map:`pcacov`.

        Parameters
        ----------
        C : ``(n_features, n_features)`` `ndarray` or `scipy.sparse`
            The Covariance/Scatter matrix. If it is a precision matrix (inverse
            covariance), then set `is_inverse=True`.
        mean : :map:`Vectorizable`
            The mean instance. It must be a :map:`Vectorizable` and *not* an
            `ndarray`.
        n_samples : `int`
            The number of samples used to generate the covariance matrix.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the covariance matrix.
        is_inverse : `bool`, optional
            It ``True``, then it is assumed that `C` is a precision matrix (
            inverse covariance). Thus, the eigenvalues will be inverted. If
            ``False``, then it is assumed that `C` is a covariance matrix.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # Create new pca instance
        self_model = PCAVectorModel.__new__(cls)
        self_model.n_samples = n_samples

        # Compute pca on covariance
        e_vectors, e_values = pcacov(C, is_inverse=is_inverse)

        # The call to __init__ of MeanLinearModel is done in here
        self_model._constructor_helper(eigenvalues=e_values,
                                       eigenvectors=e_vectors,
                                       mean=mean.as_vector(),
                                       centred=centred,
                                       max_n_components=max_n_components)
        VectorizableBackedModel.__init__(self_model, mean)
        return self_model

    @classmethod
    def init_from_components(cls, components, eigenvalues, mean, n_samples,
                             centred, max_n_components=None):
        r"""
        Build the Principal Component Analysis (PCA) using the provided
        components (eigenvectors) and eigenvalues.

        Parameters
        ----------
        components : ``(n_components, n_features)`` `ndarray`
            The eigenvectors to be used.
        eigenvalues : ``(n_components, )`` `ndarray`
            The corresponding eigenvalues.
        mean : :map:`Vectorizable`
            The mean instance. It must be a :map:`Vectorizable` and *not* an
            `ndarray`.
        n_samples : `int`
            The number of samples used to generate the eigenvectors.
        centred : `bool`, optional
            When ``True`` we assume that the data were centered before
            computing the eigenvectors.
        max_n_components : `int`, optional
            The maximum number of components to keep in the model. Any
            components above and beyond this one are discarded.
        """
        # Create new pca instance
        self_model = PCAVectorModel.__new__(cls)
        self_model.n_samples = n_samples

        # The call to __init__ of MeanLinearModel is done in here
        self_model._constructor_helper(
            eigenvalues=eigenvalues, eigenvectors=components,
            mean=mean.as_vector(), centred=centred,
            max_n_components=max_n_components)
        VectorizableBackedModel.__init__(self_model, mean)
        return self_model

    def mean(self):
        r"""
        Return the mean of the model.

        :type: :map:`Vectorizable`
        """
        return self.template_instance.from_vector(self._mean)

    @property
    def mean_vector(self):
        r"""
        Return the mean of the model as a 1D vector.

        :type: `ndarray`
        """
        return self._mean

    @doc_inherit(name='project_out')
    def project_out_vector(self, instance_vector):
        return PCAVectorModel.project_out(self, instance_vector)

    @doc_inherit(name='reconstruct')
    def reconstruct_vector(self, instance_vector):
        return PCAVectorModel.reconstruct(self, instance_vector)

    @doc_inherit(name='project')
    def project_vector(self, instance_vector):
        return PCAVectorModel.project(self, instance_vector)

    @doc_inherit(name='instance')
    def instance_vector(self, weights, normalized_weights=False):
        return PCAVectorModel.instance(self, weights,
                                       normalized_weights=normalized_weights)

    @doc_inherit(name='component')
    def component_vector(self, index, with_mean=True, scale=1.0):
        return PCAVectorModel.component(self, index, with_mean=with_mean,
                                        scale=scale)

    @doc_inherit(name='project_whitened')
    def project_whitened_vector(self, vector_instance):
        return PCAVectorModel.project_whitened(self, vector_instance)

    def component(self, index, with_mean=True, scale=1.0):
        r"""
        Return a particular component of the linear model.

        Parameters
        ----------
        index : `int`
            The component that is to be returned
        with_mean: `bool`, optional
            If ``True``, the component will be blended with the mean vector
            before being returned. If not, the component is returned on it's
            own.
        scale : `float`, optional
            A scale factor that should be applied to the component. Only
            valid in the case where ``with_mean == True``. See
            :meth:`component_vector` for how this scale factor is interpreted.

        Returns
        -------
        component : `type(self.template_instance)`
            The requested component instance.
        """
        return self.template_instance.from_vector(self.component_vector(
            index, with_mean=with_mean, scale=scale))

    def instance(self, weights, normalized_weights=False):
        """
        Creates a new instance of the model using the first ``len(weights)``
        components.

        Parameters
        ----------
        weights : ``(n_weights,)`` `ndarray` or `list`
            ``weights[i]`` is the linear contribution of the i'th component
            to the instance vector.
        normalized_weights : `bool`, optional
            If ``True``, the weights are assumed to be normalized w.r.t the
            eigenvalues. This can be easier to create unique instances by
            making the weights more interpretable.
        Raises
        ------
        ValueError
            If n_weights > n_components

        Returns
        -------
        instance : `type(self.template_instance)`
            An instance of the model.
        """
        v = self.instance_vector(weights, normalized_weights=normalized_weights)
        return self.template_instance.from_vector(v)

    def project_whitened(self, instance):
        """
        Projects the `instance` onto the whitened components, retrieving the 
        whitened linear weightings.

        Parameters
        ----------
        instance : :map:`Vectorizable`
            A novel instance.

        Returns
        -------
        projected : (n_components,)
            A vector of whitened linear weightings
        """
        return self.project_whitened_vector(instance.as_vector())

    def increment(self, samples, n_samples=None, forgetting_factor=1.0,
                  verbose=False):
        r"""
        Update the eigenvectors, eigenvalues and mean vector of this model
        by performing incremental PCA on the given samples.

        Parameters
        ----------
        samples : `list` of :map:`Vectorizable`
            List of new samples to update the model from.
        n_samples : `int`, optional
            If provided then ``samples``  must be an iterator that yields
            ``n_samples``. If not provided then samples has to be a
            list (so we know how large the data matrix needs to be).
        forgetting_factor : ``[0.0, 1.0]`` `float`, optional
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
        # build a data matrix from the new samples
        data = as_matrix(samples, length=n_samples, verbose=verbose)
        n_new_samples = data.shape[0]
        PCAVectorModel.increment(self, data, n_samples=n_new_samples,
                                 forgetting_factor=forgetting_factor,
                                 verbose=verbose)

    def __str__(self):
        str_out = 'PCA Model \n'                             \
                  ' - instance class:       {}\n'            \
                  ' - centred:              {}\n'            \
                  ' - # features:           {}\n'            \
                  ' - # active components:  {}\n'            \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n'            \
                  ' - components shape:     {}\n'.format(
            type(self.template_instance), self.centred,  self.n_features,
            self.n_active_components, self.variance(), self.variance_ratio(),
            self.noise_variance(), self.noise_variance_ratio(),
            self.n_components, self.components.shape)
        return str_out
