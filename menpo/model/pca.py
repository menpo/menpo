from __future__ import division
import numpy as np
from menpo.math import pca, ipca, as_matrix
from menpo.model.base import MeanInstanceLinearModel


class PCAModel(MeanInstanceLinearModel):
    r"""
    A :map:`MeanInstanceLinearModel` where components are Principal
    Components.

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
     """
    def __init__(self, samples, centre=True, n_samples=None, verbose=False):
        # build a data matrix from all the samples
        data, template = as_matrix(samples, length=n_samples,
                                   return_template=True, verbose=verbose)
        # (n_samples, n_features)
        self.n_samples = data.shape[0]

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

        :type: `int`
        """
        return self._n_active_components

    @n_active_components.setter
    def n_active_components(self, value):
        r"""
        Sets an updated number of active components on this model.

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

    @MeanInstanceLinearModel.components.getter
    def components(self):
        r"""
        Returns the active components of the model.

        :type: ``(n_active_components, n_features)`` `ndarray`
        """
        return self._components[:self.n_active_components, :]

    @property
    def eigenvalues(self):
        r"""
        Returns the eigenvalues associated to the active components of the
        model, i.e. the amount of variance captured by each active component.

        :type: ``(n_active_components,)`` `ndarray`
        """
        return self._eigenvalues[:self.n_active_components]

    def whitened_components(self):
        r"""
        Returns the active components of the model whitened.

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
        if noise_variance == 0:
            raise ValueError("noise variance is 0 - cannot take the inverse")
        return 1.0 / noise_variance

    def component_vector(self, index, with_mean=True, scale=1.0):
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
            return (scaled_eigval * self.components[index]) + self.mean_vector
        else:
            return self.components[index]

    def instance_vectors(self, weights):
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
            full_weights = np.zeros((n_instances, self.n_active_components))
            full_weights[..., :n_weights] = weights
            weights = full_weights
        return self._instance_vectors_for_full_weights(weights)

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

    def project_whitened_vector(self, vector_instance):
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
        # (n_samples, n_features)
        n_new_samples = data.shape[0]

        # compute incremental pca
        e_vectors, e_values, m_vector = ipca(
            data, self._components, self._eigenvalues, self.n_samples,
            m_a=self.mean_vector, f=forgetting_factor)

        # if the number of active components is the same as the total number
        # of components so it will be after this method is executed
        reset = (self.n_active_components == self.n_components)

        # update mean, components, eigenvalues and number of samples
        self.mean_vector = m_vector
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
            The size of the markers in points^2.
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
        from menpo.visualize import GraphPlotter
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(self.n_active_components),
                            y_axis=[self.eigenvalues], title='Eigenvalues',
                            x_label='Component Number', y_label='Eigenvalue',
                            x_axis_limits=(0, self.n_active_components - 1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

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
            The size of the markers in points^2.
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
        from menpo.visualize import GraphPlotter
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(self.n_active_components),
                            y_axis=[self.eigenvalues_ratio()],
                            title='Variance Ratio of Eigenvalues',
                            x_label='Component Number',
                            y_label='Variance Ratio',
                            x_axis_limits=(0, self.n_active_components - 1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

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
            The size of the markers in points^2.
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
        from menpo.visualize import GraphPlotter
        return GraphPlotter(figure_id=figure_id, new_figure=new_figure,
                            x_axis=range(self.n_active_components),
                            y_axis=[self.eigenvalues_cumulative_ratio()],
                            title='Cumulative Variance Ratio of Eigenvalues',
                            x_label='Component Number',
                            y_label='Cumulative Variance Ratio',
                            x_axis_limits=(0, self.n_active_components - 1),
                            y_axis_limits=None).render(
            render_lines=render_lines, line_colour=line_colour,
            line_style=line_style, line_width=line_width,
            render_markers=render_markers, marker_style=marker_style,
            marker_size=marker_size, marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_legend=False,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, render_grid=render_grid,
            grid_line_style=grid_line_style, grid_line_width=grid_line_width,
            figure_size=figure_size)

    def __str__(self):
        str_out = 'PCA Model \n'                             \
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
