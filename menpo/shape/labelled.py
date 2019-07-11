import warnings
from collections import OrderedDict

import numpy as np

from menpo.base import Copyable
from menpo.shape import PointUndirectedGraph, PointCloud, TriMesh
from menpo.shape.graph import (_convert_edges_to_symmetric_adjacency_matrix,
                               PointGraph)
from menpo.visualize import viewwrapper


def indices_to_masks(labels_to_indices, n_points):
    r"""
    Take a dictionary of labels to indices and convert it to a dictionary
    that maps labels to masks. This dictionary is the correct format for
    constructing a :map:`LabelledPointUndirectedGraph`.

    Parameters
    ----------
    labels_to_indices : `ordereddict` {`str` -> `int ndarray`}
        For each label, the indices in to the pointcloud that belong to the
        label.
    n_points : `int`
        Number of points in the pointcloud that is being masked.
    """
    if not isinstance(labels_to_indices, OrderedDict):
        raise ValueError('Must provide an OrderedDict to maintain the '
                         'semantic meaning of the labels.')

    masks = OrderedDict()
    for label in labels_to_indices:
        indices = labels_to_indices[label]
        mask = np.zeros(n_points, dtype=np.bool)
        mask[indices] = True
        masks[label] = mask
    return masks


class LabelledPointUndirectedGraph(PointUndirectedGraph):
    r"""
    A subclass of :map:`PointUndirectedGraph` that allows the attaching
    of 'labels' associated with semantic parts of an object. For example,
    for a face the semantic parts might be the eyes, nose and mouth. These
    'labels' are defined as a dictionary of string keys that map to
    boolean mask arrays that define which of the underlying points belong
    to a given label.

    The labels to masks must be within an `OrderedDict` so that semantic
    ordering can be maintained.

    Parameters
    ----------
    points : `ndarray`
        The points representing the landmarks.
    adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
        The adjacency matrix of the graph. The non-edges must be represented
        with zeros and the edges can have a weight value.

        :Note: ``adjacency_matrix`` must be symmetric.
    labels_to_masks : `ordereddict` {`str` -> `bool ndarray`}
        For each label, the mask that specifies the indices in to the
        points that belong to the label.
    copy : `bool`, optional
        If ``True``, a copy of the data is stored.

    Raises
    ------
    ValueError
        If `dict` passed instead of `OrderedDict`
    ValueError
        If no set of label masks is passed.
    ValueError
        If any of the label masks differs in size to the points.
    ValueError
        If there exists any point in the points that is not covered
        by a label.
    """
    def __init__(self, points, adjacency_matrix, labels_to_masks, copy=True,
                 skip_checks=False):
        PointUndirectedGraph.__init__(self, points, adjacency_matrix, copy=copy,
                                      skip_checks=skip_checks)

        if not labels_to_masks:
            raise ValueError('Labelled point graphs are designed to be '
                             'immutable. Empty label sets are not permitted.')
        if np.vstack(list(labels_to_masks.values())).shape[1] != points.shape[0]:
            raise ValueError('Each mask must have the same number of points '
                             'as the given points.')
        if not isinstance(labels_to_masks, OrderedDict):
            raise ValueError('Must provide an OrderedDict to maintain the '
                             'semantic meaning of the labels.')

        # Another sanity check
        self._labels_to_masks = labels_to_masks
        self._verify_all_labels_masked()

        if copy:
            self._labels_to_masks = OrderedDict([(l, m.copy()) for l, m in
                                                 labels_to_masks.items()])

    @classmethod
    def init_with_all_label(cls, points, adjacency_matrix, copy=True):
        r"""
        Static constructor to create a :map:`LabelledPointUndirectedGraph` with
        a single default 'all' label that covers all points.

        Parameters
        ----------
        points : `ndarray`
            The points representing the landmarks.
        adjacency_matrix : ``(n_vertices, n_vertices, )`` `ndarray` or `csr_matrix`
            The adjacency matrix of the graph. The non-edges must be represented
            with zeros and the edges can have a weight value.

            :Note: ``adjacency_matrix`` must be symmetric.
        copy : `bool`, optional
            If ``True``, a copy of data is stored on the group.

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            Labelled pointgraph wrapping the given points with a single label
            called 'all' that is ``True`` for all points.
        """
        labels_to_masks = OrderedDict(
            [('all', np.ones(points.shape[0], dtype=np.bool))])
        return LabelledPointUndirectedGraph(points, adjacency_matrix,
                                            labels_to_masks, copy=copy)

    @classmethod
    def init_from_indices_mapping(cls, points, adjacency,
                                  labels_to_indices, copy=True):
        r"""
        Static constructor to create a :map:`LabelledPointUndirectedGraph` from
        an ordered dictionary that maps a set of indices .

        Parameters
        ----------
        points : :map:`PointCloud`
            The points representing the landmarks.
        adjacency : ``(n_vertices, n_vertices, )`` `ndarray`, `csr_matrix` or `list` of edges
            The adjacency matrix of the graph, or a list of edges representing
            adjacency.
        labels_to_indices : `ordereddict` {`str` -> `int ndarray`}
            For each label, the indices in to the points that belong to the
            label.
        copy : `boolean`, optional
            If ``True``, a copy of the data is stored on the group.

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            Labelled point undirected graph wrapping the given points with the
            given semantic labels applied.

        Raises
        ------
        ValueError
            If `dict` passed instead of `OrderedDict`
        ValueError
            If any of the label masks differs in size to the points.
        ValueError
            If there exists any point in the points that is not covered
            by a label.
        """
        adjacency = np.array(adjacency)
        if adjacency.shape[0] != adjacency.shape[1] and adjacency.shape[1] == 2:
            adjacency = _convert_edges_to_symmetric_adjacency_matrix(
                adjacency, points.shape[0])
        labels_to_masks = indices_to_masks(labels_to_indices,
                                           points.shape[0])
        return LabelledPointUndirectedGraph(points, adjacency, labels_to_masks,
                                            copy=copy)

    @classmethod
    def init_from_edges(cls, points, edges, labels_to_masks, copy=True,
                        skip_checks=False):
        r"""
        Construct a :map:`LabelledPointUndirectedGraph` from an edges array.

        See :map:`PointUndirectedGraph` for more information.

        Parameters
        ----------
        points : ``(n_vertices, n_dims, )`` `ndarray`
            The array of point locations.
        edges : ``(n_edges, 2, )`` `ndarray` or ``None``
            The `ndarray` of edges, i.e. all the pairs of vertices that are
            connected with an edge. If ``None``, then an empty adjacency
            matrix is created.
        labels_to_masks : `ordereddict` `{str -> bool ndarray}`
            For each label, the mask that specifies the indices in to the
            points that belong to the label.
        copy : `bool`, optional
            If ``False``, the `adjacency_matrix` will not be copied on
            assignment.
        skip_checks : `bool`, optional
            If ``True``, no checks will be performed.
        """
        adjacency_matrix = _convert_edges_to_symmetric_adjacency_matrix(
            edges, points.shape[0])
        return cls(points, adjacency_matrix, labels_to_masks, copy=copy,
                   skip_checks=skip_checks)

    def __setstate__(self, state_dict):
        # TODO: Deprecate this - this handles importing old-style LandmarkGroup
        if '_pointcloud' in state_dict:
            from menpo.base import MenpoDeprecationWarning
            warnings.warn('menpo.landmark.LandmarkGroup is now deprecated and '
                          'has been moved to menpo.shape.LandmarkGroup.',
                          MenpoDeprecationWarning)
            _pointcloud = state_dict.pop('_pointcloud')
            state_dict['points'] = _pointcloud.points

            # the shape on old landmarks *itself* was allowed to have landmarks
            # (of course it was very frequently None though, see
            # https://github.com/menpo/menpo/blob/v0.7.7/menpo/landmark/base.py#L24)
            # In the new word, self has the same behavior, so move the
            # landmarks across here.
            # In the vast majority of cases, this will simply be None.
            state_dict['_landmarks'] = _pointcloud._landmarks

            if type(_pointcloud) == PointCloud:
                adj_mat = _convert_edges_to_symmetric_adjacency_matrix(
                    [], _pointcloud.n_points)
            elif isinstance(_pointcloud, PointGraph):
                a = _pointcloud.adjacency_matrix
                # Ensure that the matrix is symmetric
                adj_mat = a.maximum(a.T)
            elif isinstance(_pointcloud, TriMesh):
                warnings.warn('menpo.landmark.LandmarkGroup is now deprecated.'
                              'The underlying ._pointcloud was a '
                              'menpo.shape.TriMesh and this has been cast down '
                              'to an UndirectedPointGraph subclass.')
                adj_mat = _pointcloud.as_pointgraph(copy=False).adjacency_matrix
            else:
                raise ValueError('Unexpected PointCloud type ({})'.format(
                    type(_pointcloud)))
            state_dict['adjacency_matrix'] = adj_mat

        self.__dict__.update(state_dict)

    def copy(self):
        r"""
        Generate an efficient copy of this :map:`LabelledPointUndirectedGraph`.

        Returns
        -------
        ``type(self)``
            A copy of this object
        """
        new = Copyable.copy(self)
        for k, v in new._labels_to_masks.items():
            new._labels_to_masks[k] = v.copy()
        return new

    def add_label(self, label, indices):
        """
        Add a new label by creating a new mask over the points. A new
        :map:`LabelledPointUndirectedGraph` is returned.

        Parameters
        ----------
        label : `string`
            Label of landmark.
        indices : ``(K,)`` `ndarray`
            Array of indices in to the points. Each index implies
            membership to the label.

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            A new labelled pointgraph with the new label specified by indices.
        """
        new = self.copy()
        mask = np.zeros(self.n_points, dtype=np.bool)
        mask[indices] = True
        new._labels_to_masks[label] = mask
        return new

    def get_label(self, label):
        """
        Returns a new :map:`PointUndirectedGraph` that contains the subset of
        points that this label represents.

        Parameters
        ----------
        label : `string`
            Label to filter on.

        Returns
        -------
        graph : :map:`PointUndirectedGraph`
            The PointUndirectedGraph containing the subset of points that this
            label masks. Will be a subset of the entire group's points.
        """
        mask = self._labels_to_masks[label]
        return PointUndirectedGraph.from_mask(self, mask)

    def remove_label(self, label):
        """
        Returns a new :map:`LabelledPointUndirectedGraph` that does not contain
        the given label.

         .. note::

             You cannot delete a semantic label and leave the labelled point
             graph partially unlabelled. Labelled point graphs must contain
             labels for **every point**.

        Parameters
        ---------
        label : `string`
            The label to remove.

        Raises
        ------
        ValueError
            If deleting the label would leave some points unlabelled.
        """
        new = self.copy()
        # Pop the value off, which is akin to deleting it (removes it from the
        # underlying dict). However, we keep it around so we can check if
        # removing it causes an unlabelled point
        new._labels_to_masks.pop(label)
        new._verify_all_labels_masked()
        return new

    @property
    def labels(self):
        """
        The list of labels that belong to this group.

        :type: `list` of `str`
        """
        # Convert to list so that we can index immediately, as keys()
        # is a view in Python 3
        return list(self._labels_to_masks.keys())

    @property
    def n_labels(self):
        """
        Number of labels in the group.

        :type: `int`
        """
        return len(self.labels)

    @property
    def n_landmarks(self):
        """
        The total number of points in the group.

        :type: `int`
        """
        from menpo.base import MenpoDeprecationWarning
        warnings.warn('The .n_landmarks property is deprecated. LandmarkGroups '
                      'are now LabelledPointUndirectedGraph which '
                      'are subclasses of UndirectedPointGraph and thus may '
                      'be used as such. Thus .n_landmarks is an alias for '
                      '.n_points .',
                      MenpoDeprecationWarning)
        return self.n_points

    def with_labels(self, labels):
        """A new labelled point undirected graph that contains only the given
        labels.

        Parameters
        ----------
        labels : `str` or `list` of `str`
            Label(s) that should be kept in the returned labelled point graph.

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            A new labelled point undirected graph with the same group label but
            containing only the given label(s).
        """
        # Make it easier to use by accepting a single string as well as a list
        if isinstance(labels, str):
            labels = [labels]
        return self._new_group_with_only_labels(labels)

    def without_labels(self, labels):
        """A new labelled point undirected graph that excludes certain labels.

        Parameters
        ----------
        labels : `str` or `list` of `str`
            Label(s) that should be excluded in the returned labelled point
            graph.

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            A new labelled point undirected graph with the same group label but
            containing all labels except the given label.
        """
        # Make it easier to use by accepting a single string as well as a list
        if isinstance(labels, str):
            labels = [labels]
        labels_to_keep = list(set(self.labels).difference(labels))
        return self._new_group_with_only_labels(labels_to_keep)

    def _verify_all_labels_masked(self):
        """
        Verify that every point in the pointcloud is associated with a label.
        If any one point is not covered by a label, then raise a
        ``ValueError``.
        """
        # values is a generator in Python 3, so convert to list
        labels_values = list(self._labels_to_masks.values())
        unlabelled_points = np.sum(labels_values, axis=0) == 0
        if np.any(unlabelled_points):
            nonzero = np.nonzero(unlabelled_points)
            raise ValueError(
                'Every point in the landmark pointcloud must be labelled. '
                'Points {0} were unlabelled.'.format(nonzero))

    def _new_group_with_only_labels(self, labels):
        """
        Deal with changing indices when you add and remove points. In this case
        we only deal with building a new dataset that keeps masks.

        Parameters
        ----------
        labels : list of `string`
            List of strings of the labels to keep

        Returns
        -------
        labelled_pointgraph : :map:`LabelledPointUndirectedGraph`
            The new labelled pointgraph with only the requested labels.
        """
        set_difference = set(labels).difference(self.labels)
        if len(set_difference) > 0:
            raise ValueError('Labels {0} do not exist in the landmark '
                             'group. Available labels are: {1}'.format(
                list(set_difference), self.labels))

        masks_to_keep = [self._labels_to_masks[l] for l in labels
                         if l in self._labels_to_masks]
        overlap = np.sum(masks_to_keep, axis=0) > 0
        masks_to_keep = [l[overlap] for l in masks_to_keep]

        new_graph = self.from_mask(overlap)
        return LabelledPointUndirectedGraph(new_graph.points,
                                            new_graph.adjacency_matrix,
                                            OrderedDict(zip(labels,
                                                            masks_to_keep)))

    def tojson(self):
        r"""
        Convert this `LabelledPointUndirectedGraph` to a dictionary JSON
        representation.

        Returns
        -------
        json : ``dict``
            Dictionary conforming to the LJSON v2 specification.
        """
        labels = [{'mask': mask.nonzero()[0].tolist(),
                   'label': label}
                  for label, mask in self._labels_to_masks.items()]
        lms_dict = PointUndirectedGraph.tojson(self)
        lms_dict['labels'] = labels
        return lms_dict

    def _view_2d(self, with_labels=None, without_labels=None, group='group',
                 figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour=None, line_style='-',
                 line_width=1, render_markers=True, marker_style='o',
                 marker_size=5, marker_face_colour=None,
                 marker_edge_colour=None, marker_edge_width=1.,
                 render_numbering=False, numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_legend=True, legend_title='',
                 legend_font_name='sans-serif', legend_font_style='normal',
                 legend_font_size=10, legend_font_weight='normal',
                 legend_marker_scale=None, legend_location=2,
                 legend_bbox_to_anchor=(1.05, 1.), legend_border_axes_pad=None,
                 legend_n_columns=1, legend_horizontal_spacing=None,
                 legend_vertical_spacing=None, legend_border=True,
                 legend_border_padding=None, legend_shadow=False,
                 legend_rounded_corners=False, render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8)):
        """
        Visualize the labelled point undirected graph.

        Parameters
        ----------
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        group : `str` or `None`, optional
            The name of the labelled point undirected graph. It is used in
            the legend.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True``, the x and y axes are flipped.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

            It can either be one of the above or a `list` of those defining a
            value per label.
        line_style : ``{'-', '--', '-.', ':'}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

            It can either be one of the above or a `list` of those defining a
            value per label.
        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

            It can either be one of the above or a `list` of those defining a
            value per label.
        marker_edge_width : `float`, optional
            The width of the markers' edge.
        render_numbering : `bool`, optional
            If ``True``, the landmarks will be numbered.
        numbers_horizontal_align : ``{center, right, left}``, optional
            The horizontal alignment of the numbers' texts.
        numbers_vertical_align : ``{center, top, bottom, baseline}``, optional
            The vertical alignment of the numbers' texts.
        numbers_font_name : See Below, optional
            The font of the numbers. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        numbers_font_size : `int`, optional
            The font size of the numbers.
        numbers_font_style : ``{normal, italic, oblique}``, optional
            The font style of the numbers.
        numbers_font_weight : See Below, optional
            The font weight of the numbers.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        numbers_font_colour : See Below, optional
            The font colour of the numbers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        render_legend : `bool`, optional
            If ``True``, the legend will be rendered.
        legend_title : `str`, optional
            The title of the legend.
        legend_font_name : See Below, optional
            The font of the legend.
            Possible options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : See Below, optional
            The font weight of the legend.
            Possible options ::

                {ultralight, light, normal, regular, book, medium, roman,
                 semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ===
            'best'          0
            'upper right'   1
            'upper left'    2
            'lower left'    3
            'lower right'   4
            'right'         5
            'center left'   6
            'center right'  7
            'lower center'  8
            'upper center'  9
            'center'        10
            =============== ===

        legend_bbox_to_anchor : (`float`, `float`), optional
            The bbox that the legend will be anchored.
        legend_border_axes_pad : `float`, optional
            The pad between the axes and legend border.
        legend_n_columns : `int`, optional
            The number of the legend's columns.
        legend_horizontal_spacing : `float`, optional
            The spacing between the columns.
        legend_vertical_spacing : `float`, optional
            The vertical space between the legend entries.
        legend_border : `bool`, optional
            If ``True``, a frame will be drawn around the legend.
        legend_border_padding : `float`, optional
            The fractional whitespace inside the legend border.
        legend_shadow : `bool`, optional
            If ``True``, a shadow will be drawn behind legend.
        legend_rounded_corners : `bool`, optional
            If ``True``, the frame's corners will be rounded (fancybox).
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes.
            Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : {``normal``, ``italic``, ``oblique``}, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the LabelledPointUndirectedGraph as a percentage
            of the LabelledPointUndirectedGraph's width. If `tuple` or `list`,
            then it defines the axis limits. If ``None``, then the limits are
            set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the LabelledPointUndirectedGraph as a percentage
            of the LabelledPointUndirectedGraph's height. If `tuple` or `list`,
            then it defines the axis limits. If ``None``, then the limits are
            set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) or `None`, optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        """
        from menpo.visualize import LandmarkViewer2d
        if with_labels is not None and without_labels is not None:
            raise ValueError('You may only pass one of `with_labels` or '
                             '`without_labels`.')
        elif with_labels is not None:
            lmark_group = self.with_labels(with_labels)
        elif without_labels is not None:
            lmark_group = self.without_labels(without_labels)
        else:
            lmark_group = self  # Fall through
        landmark_viewer = LandmarkViewer2d(figure_id, new_figure,
                                           group, lmark_group)
        return landmark_viewer.render(
            image_view=image_view, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width,
            render_numbering=render_numbering,
            numbers_horizontal_align=numbers_horizontal_align,
            numbers_vertical_align=numbers_vertical_align,
            numbers_font_name=numbers_font_name,
            numbers_font_size=numbers_font_size,
            numbers_font_style=numbers_font_style,
            numbers_font_weight=numbers_font_weight,
            numbers_font_colour=numbers_font_colour,
            render_legend=render_legend, legend_title=legend_title,
            legend_font_name=legend_font_name,
            legend_font_style=legend_font_style,
            legend_font_size=legend_font_size,
            legend_font_weight=legend_font_weight,
            legend_marker_scale=legend_marker_scale,
            legend_location=legend_location,
            legend_bbox_to_anchor=legend_bbox_to_anchor,
            legend_border_axes_pad=legend_border_axes_pad,
            legend_n_columns=legend_n_columns,
            legend_horizontal_spacing=legend_horizontal_spacing,
            legend_vertical_spacing=legend_vertical_spacing,
            legend_border=legend_border,
            legend_border_padding=legend_border_padding,
            legend_shadow=legend_shadow,
            legend_rounded_corners=legend_rounded_corners,
            render_axes=render_axes, axes_font_name=axes_font_name,
            axes_font_size=axes_font_size, axes_font_style=axes_font_style,
            axes_font_weight=axes_font_weight, axes_x_limits=axes_x_limits,
            axes_y_limits=axes_y_limits, axes_x_ticks=axes_x_ticks,
            axes_y_ticks=axes_y_ticks, figure_size=figure_size)

    def _view_3d(self, with_labels=None, without_labels=None, group='group',
                 figure_id=None, new_figure=False, render_lines=True,
                 line_colour=None, line_width=2, render_markers=True,
                 marker_style='sphere', marker_size=None, marker_colour=None,
                 marker_resolution=8, step=None, alpha=1.0,
                 render_numbering=False, numbers_colour='k', numbers_size=None):
        try:
            from menpo3d.visualize import LandmarkViewer3d
            if with_labels is not None and without_labels is not None:
                raise ValueError('You may only pass one of `with_labels` or '
                                 '`without_labels`.')
            elif with_labels is not None:
                lmark_group = self.with_labels(with_labels)
            elif without_labels is not None:
                lmark_group = self.without_labels(without_labels)
            else:
                lmark_group = self  # Fall through
            landmark_viewer = LandmarkViewer3d(figure_id, new_figure,
                                               group, lmark_group)
            return landmark_viewer.render(
                render_lines=render_lines, line_colour=line_colour,
                line_width=line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_colour=marker_colour, marker_resolution=marker_resolution,
                step=step, alpha=alpha, render_numbering=render_numbering,
                numbers_colour=numbers_colour, numbers_size=numbers_size)
        except ImportError as e:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError(e)

    @viewwrapper
    def view_widget(self, ):
        r"""
        Abstract method for viewing with an interactive widget. See the
        :map:`viewwrapper` documentation for an explanation of how the
        `view_widget` method works.
        """
        pass

    def _view_widget_2d(self, figure_size=(7, 7)):
        r"""
        Visualization of the LabelledPointUndirectedGraph using an interactive 
        widget.

        Parameters
        ----------
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        """
        try:
            from menpowidgets import view_widget
            view_widget(self, figure_size=figure_size)
        except ImportError as e:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError(e)

    def _view_widget_3d(self):
        r"""
        Visualization of the LabelledPointUndirectedGraph using an interactive 
        widget.
        """
        try:
            from menpowidgets import view_widget
            view_widget(self)
        except ImportError as e:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError(e)

    def __str__(self):
        return '{}: n_labels: {}, n_points: {}, n_edges: {}'.format(
            type(self).__name__, self.n_labels, self.n_points,
            self.n_edges
        )
