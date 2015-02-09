from __future__ import division
import numpy as np
from menpo.math import principal_component_decomposition
from menpo.model.base import MeanInstanceLinearModel
from menpo.visualize import print_dynamic, progress_bar_str


class PCAModel(MeanInstanceLinearModel):
    r"""
    A :map:`MeanInstanceLinearModel` where components are Principal
    Components.

    Principal Component Analysis (PCA) by eigenvalue decomposition of the
    data's scatter matrix. For details of the implementation of PCA, see
    :map:`principal_component_decomposition`.

    Parameters
    ----------
    samples : `list` of :map:`Vectorizable`
        List of samples to build the model from.
    centre : `bool`, optional
        When ``True`` (default) PCA is performed after mean centering the data.
        If ``False`` the data is assumed to be centred, and the mean will be
        ``0``.
    bias : `bool`, optional
        When ``True`` a biased estimator of the covariance matrix is used.
        See notes.
    n_samples : `int`, optional
        If provided then ``samples``  must be an iterator  that yields
        ``n_samples``. If not provided then samples has to be a `list` (so we
        know how large the data matrix needs to be).

    Notes
    -----
    True bias means that we calculate the covariance as
    :math:`\frac{1}{N} \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T` instead of
    default :math:`\frac{1}{N-1} \sum_{i=1}^N \mathbf{x}_i \mathbf{x}_i^T`.
    """
    def __init__(self, samples, centre=True, bias=False, verbose=False,
                 n_samples=None):
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

        # compute pca
        e_vectors, e_values, mean = principal_component_decomposition(
            data, whiten=False,  centre=centre, bias=bias, inplace=True)

        super(PCAModel, self).__init__(e_vectors, mean, template)
        self.centred = centre
        self.biased = bias
        self._eigenvalues = e_values
        # start the active components as all the components
        self._n_active_components = int(self.n_components)
        self._trimmed_eigenvalues = None

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
            np.sqrt(self.eigenvalues + self.noise_variance())[:, None])

    def original_variance(self):
        r"""
        Returns the total amount of variance captured by the original model,
        i.e. the amount of variance present on the original samples.

        Returns
        -------
        optional_variance : `float`
            The variance captured by the model.
        """
        original_variance = self._eigenvalues.sum()
        if self._trimmed_eigenvalues is not None:
            original_variance += self._trimmed_eigenvalues.sum()
        return original_variance

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
        instance : :map:`Vectorizable`
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
        vector_instance : ``(n_features,)`` `ndarray`
            A novel vector.

        Returns
        -------
        scaled_projected_out : ``(n_features,)`` `ndarray`
            A copy of `vector_instance` with all basis of the model projected
            out and scaled by the inverse of the `noise_variance`.
        """
        return (self.inverse_noise_variance() *
                self.project_out_vectors(vector_instance))

    def project_whitened(self, instance):
        """
        Returns a sheared (non-orthogonal) reconstruction of `instance`.

        Parameters
        ----------
        instance : :map:`Vectorizable`
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
        Returns a sheared (non-orthogonal) reconstruction of `vector_instance`.

        Parameters
        ----------
        vector_instance : ``(n_features,)`` `ndarray`
            A novel vector.

        Returns
        -------
        sheared_reconstruction : ``(n_features,)`` `ndarray`
            A sheared (non-orthogonal) reconstruction of `vector_instance`
        """
        whitened_components = self.whitened_components()
        weights = np.dot(vector_instance, whitened_components.T)
        return np.dot(weights, whitened_components)

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
                  ' - biased:               {}\n'            \
                  ' - # features:           {}\n'            \
                  ' - # active components:  {}\n'            \
                  ' - kept variance:        {:.2}  {:.1%}\n' \
                  ' - noise variance:       {:.2}  {:.1%}\n' \
                  ' - total # components:   {}\n'            \
                  ' - components shape:     {}\n'.format(
            self.centred,  self.biased, self.n_features,
            self.n_active_components, self.variance(), self.variance_ratio(),
            self.noise_variance(), self.noise_variance_ratio(),
            self.n_components, self.components.shape)
        return str_out
