import numpy as np
import numbers
import collections
from warnings import warn
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from menpo.shape.base import Shape


def bounding_box(closest_to_origin, opposite_corner):
    r"""
    Return a bounding box from two corner points as a directed graph.
    The the first point (0) should be nearest the origin.
    In the case of an image, this ordering would appear as:

    ::

        0<--3
        |   ^
        |   |
        v   |
        1-->2

    In the case of a pointcloud, the ordering will appear as:

    ::

        3<--2
        |   ^
        |   |
        v   |
        0-->1


    Parameters
    ----------
    closest_to_origin : (`float`, `float`)
        Two floats representing the coordinates closest to the origin.
        Represented by (0) in the graph above. For an image, this will
        be the top left. For a pointcloud, this will be the bottom left.
    opposite_corner  : (`float`, `float`)
        Two floats representing the coordinates opposite the corner closest
        to the origin.
        Represented by (2) in the graph above. For an image, this will
        be the bottom right. For a pointcloud, this will be the top right.

    Returns
    -------
    bounding_box : :map:`PointDirectedGraph`
        The axis aligned bounding box from the two given corners.
    """
    from .graph import PointDirectedGraph

    if len(closest_to_origin) != 2 or len(opposite_corner) != 2:
        raise ValueError('Only 2D bounding boxes can be created.')

    adjacency_matrix = csr_matrix(([1] * 4, ([0, 1, 2, 3], [1, 2, 3, 0])),
                                  shape=(4, 4))
    box = np.array([closest_to_origin,
                    [opposite_corner[0], closest_to_origin[1]],
                    opposite_corner,
                    [closest_to_origin[0], opposite_corner[1]]], dtype=np.float)
    return PointDirectedGraph(box, adjacency_matrix, copy=False)


class PointCloud(Shape):
    r"""
    An N-dimensional point cloud. This is internally represented as an `ndarray`
    of shape ``(n_points, n_dims)``. This class is important for dealing
    with complex functionality such as viewing and representing metadata such
    as landmarks.

    Currently only 2D and 3D pointclouds are viewable.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The array representing the points.
    copy : `bool`, optional
        If ``False``, the points will not be copied on assignment. Note that
        this will miss out on additional checks. Further note that we still
        demand that the array is C-contiguous - if it isn't, a copy will be
        generated anyway.
        In general this should only be used if you know what you are doing.
    """

    def __init__(self, points, copy=True):
        super(PointCloud, self).__init__()
        if not copy:
            if not points.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                points = np.array(points, copy=True, order='C')
        else:
            points = np.array(points, copy=True, order='C')
        self.points = points

    @classmethod
    def init_2d_grid(cls, shape, spacing=None):
        r"""
        Create a pointcloud that exists on a regular 2D grid. The first
        dimension is the number of rows in the grid and the second dimension
        of the shape is the number of columns. ``spacing`` optionally allows
        the definition of the distance between points (uniform over points).
        The spacing may be different for rows and columns.

        Parameters
        ----------
        shape : `tuple` of 2 `int`
            The size of the grid to create, this defines the number of points
            across each dimension in the grid. The first element is the number
            of rows and the second is the number of columns.
        spacing : `int` or `tuple` of 2 `int`, optional
            The spacing between points. If a single `int` is provided, this
            is applied uniformly across each dimension. If a `tuple` is
            provided, the spacing is applied non-uniformly as defined e.g.
            ``(2, 3)`` gives a spacing of 2 for the rows and 3 for the
            columns.

        Returns
        -------
        shape_cls : `type(cls)`
            A PointCloud or subclass arranged in a grid.
        """
        if len(shape) != 2:
            raise ValueError('shape must be 2D.')

        grid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                           indexing='ij')
        points = np.require(np.concatenate(grid).reshape([2, -1]).T,
                            dtype=np.float64, requirements=['C'])

        if spacing is not None:
            if not (isinstance(spacing, numbers.Number) or
                    isinstance(spacing, collections.Sequence)):
                raise ValueError('spacing must be either a single number '
                                 'to be applied over each dimension, or a 2D '
                                 'sequence of numbers.')
            if isinstance(spacing, collections.Sequence) and len(spacing) != 2:
                raise ValueError('spacing must be 2D.')

            points *= np.asarray(spacing, dtype=np.float64)
        return cls(points, copy=False)

    @property
    def n_points(self):
        r"""
        The number of points in the pointcloud.

        :type: `int`
        """
        return self.points.shape[0]

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the pointcloud.

        :type: `int`
        """
        return self.points.shape[1]

    def h_points(self):
        r"""
        Convert poincloud to a homogeneous array: ``(n_dims + 1, n_points)``

        :type: ``type(self)``
        """
        return np.concatenate((self.points.T,
                               np.ones(self.n_points,
                                       dtype=self.points.dtype)[None, :]))

    def centre(self):
        r"""
        The mean of all the points in this PointCloud (centre of mass).

        Returns
        -------
        centre : ``(n_dims)`` `ndarray`
            The mean of this PointCloud's points.
        """
        return np.mean(self.points, axis=0)

    def centre_of_bounds(self):
        r"""
        The centre of the absolute bounds of this PointCloud. Contrast with
        :meth:`centre`, which is the mean point position.

        Returns
        -------
        centre : ``n_dims`` `ndarray`
            The centre of the bounds of this PointCloud.
        """
        min_b, max_b = self.bounds()
        return (min_b + max_b) / 2.0

    def _as_vector(self):
        r"""
        Returns a flattened representation of the pointcloud.
        Note that the flattened representation is of the form
        ``[x0, y0, x1, y1, ....., xn, yn]`` for 2D.

        Returns
        -------
        flattened : ``(n_points,)`` `ndarray`
            The flattened points.
        """
        return self.points.ravel()

    def tojson(self):
        r"""
        Convert this :map:`PointCloud` to a dictionary representation suitable
        for inclusion in the LJSON landmark format.

        Returns
        -------
        json : `dict`
            Dictionary with ``points`` keys.
        """
        return {'points': self.points.tolist()}

    def _from_vector_inplace(self, vector):
        r"""
        Updates the points of this PointCloud in-place with the reshaped points
        from the provided vector. Note that the vector should have the form
        ``[x0, y0, x1, y1, ....., xn, yn]`` for 2D.

        Parameters
        ----------
        vector : ``(n_points,)`` `ndarray`
            The vector from which to create the points' array.
        """
        self.points = vector.reshape([-1, self.n_dims])

    def __str__(self):
        return '{}: n_points: {}, n_dims: {}'.format(type(self).__name__,
                                                     self.n_points,
                                                     self.n_dims)

    def bounds(self, boundary=0):
        r"""
        The minimum to maximum extent of the PointCloud. An optional boundary
        argument can be provided to expand the bounds by a constant margin.

        Parameters
        ----------
        boundary : `float`
            A optional padding distance that is added to the bounds. Default
            is ``0``, meaning the max/min of tightest possible containing
            square/cube/hypercube is returned.

        Returns
        -------
        min_b : ``(n_dims,)`` `ndarray`
            The minimum extent of the :map:`PointCloud` and boundary along
            each dimension
        max_b : ``(n_dims,)`` `ndarray`
            The maximum extent of the :map:`PointCloud` and boundary along
            each dimension
        """
        min_b = np.min(self.points, axis=0) - boundary
        max_b = np.max(self.points, axis=0) + boundary
        return min_b, max_b

    def range(self, boundary=0):
        r"""
        The range of the extent of the PointCloud.

        Parameters
        ----------
        boundary : `float`
            A optional padding distance that is used to extend the bounds
            from which the range is computed. Default is ``0``, no extension
            is performed.

        Returns
        -------
        range : ``(n_dims,)`` `ndarray`
            The range of the :map:`PointCloud` extent in each dimension.
        """
        min_b, max_b = self.bounds(boundary)
        return max_b - min_b

    def bounding_box(self):
        r"""
        Return a bounding box from two corner points as a directed graph.
        The the first point (0) should be nearest the origin.
        In the case of an image, this ordering would appear as:

        ::

            0<--3
            |   ^
            |   |
            v   |
            1-->2

        In the case of a pointcloud, the ordering will appear as:

        ::

            3<--2
            |   ^
            |   |
            v   |
            0-->1

        Returns
        -------
        bounding_box : :map:`PointDirectedGraph`
            The axis aligned bounding box of the PointCloud.
        """
        if self.n_dims != 2:
            raise ValueError('Bounding boxes are only supported for 2D '
                             'pointclouds.')
        min_p, max_p = self.bounds()
        return bounding_box(min_p, max_p)

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_markers=True, marker_style='o', marker_size=20,
                 marker_face_colour='r', marker_edge_colour='k',
                 marker_edge_width=1., render_numbering=False,
                 numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8), label=None, **kwargs):
        r"""
        Visualization of the PointCloud in 2D.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the PointCloud will be viewed as if it is in the image
            coordinate system.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

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
            right and left of the PointCloud as a percentage of the PointCloud's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the PointCloud as a percentage of the PointCloud's
            height. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None``, optional
            The size of the figure in inches.
        label : `str`, optional
            The name entry in case of a legend.

        Returns
        -------
        viewer : :map:`PointGraphViewer2d`
            The viewer object.
        """
        from menpo.visualize.base import PointGraphViewer2d
        adjacency_array = np.empty(0)
        renderer = PointGraphViewer2d(figure_id, new_figure,
                                      self.points,
                                      adjacency_array)
        renderer.render(
            image_view=image_view, render_lines=False, line_colour='b',
            line_style='-', line_width=1., render_markers=render_markers,
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
            numbers_font_colour=numbers_font_colour, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            figure_size=figure_size, label=label)
        return renderer

    def _view_landmarks_2d(self, group=None, with_labels=None,
                           without_labels=None, figure_id=None,
                           new_figure=False, image_view=True, render_lines=True,
                           line_colour=None, line_style='-', line_width=1,
                           render_markers=True, marker_style='o',
                           marker_size=20, marker_face_colour=None,
                           marker_edge_colour=None, marker_edge_width=1.,
                           render_numbering=False,
                           numbers_horizontal_align='center',
                           numbers_vertical_align='bottom',
                           numbers_font_name='sans-serif', numbers_font_size=10,
                           numbers_font_style='normal',
                           numbers_font_weight='normal',
                           numbers_font_colour='k', render_legend=False,
                           legend_title='', legend_font_name='sans-serif',
                           legend_font_style='normal', legend_font_size=10,
                           legend_font_weight='normal',
                           legend_marker_scale=None, legend_location=2,
                           legend_bbox_to_anchor=(1.05, 1.),
                           legend_border_axes_pad=None, legend_n_columns=1,
                           legend_horizontal_spacing=None,
                           legend_vertical_spacing=None, legend_border=True,
                           legend_border_padding=None, legend_shadow=False,
                           legend_rounded_corners=False, render_axes=False,
                           axes_font_name='sans-serif', axes_font_size=10,
                           axes_font_style='normal', axes_font_weight='normal',
                           axes_x_limits=None, axes_y_limits=None,
                           axes_x_ticks=None, axes_y_ticks=None,
                           figure_size=(10, 8)):
        """
        Visualize the landmarks. This method will appear on the Image as
        ``view_landmarks`` if the Image is 2D.

        Parameters
        ----------
        group : `str` or``None`` optional
            The landmark group to be visualized. If ``None`` and there are more
            than one landmark groups, an error is raised.
        with_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, only show the given label(s). Should **not** be
            used with the ``without_labels`` kwarg.
        without_labels : ``None`` or `str` or `list` of `str`, optional
            If not ``None``, show all except the given label(s). Should **not**
            be used with the ``with_labels`` kwarg.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the PointCloud will be viewed as if it is in the image
            coordinate system.
        render_lines : `bool`, optional
            If ``True``, the edges will be rendered.
        line_colour : See Below, optional
            The colour of the lines.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        line_style : ``{-, --, -., :}``, optional
            The style of the lines.
        line_width : `float`, optional
            The width of the lines.
        render_markers : `bool`, optional
            If ``True``, the markers will be rendered.
        marker_style : See Below, optional
            The style of the markers. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the markers in points^2.
        marker_face_colour : See Below, optional
            The face (filling) colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the markers.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

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
        legend_font_name : See below, optional
            The font of the legend. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        legend_font_style : ``{normal, italic, oblique}``, optional
            The font style of the legend.
        legend_font_size : `int`, optional
            The font size of the legend.
        legend_font_weight : See Below, optional
            The font weight of the legend.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold, demibold, demi, bold, heavy, extra bold, black}

        legend_marker_scale : `float`, optional
            The relative size of the legend markers with respect to the original
        legend_location : `int`, optional
            The location of the legend. The predefined values are:

            =============== ==
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
            =============== ==

        legend_bbox_to_anchor : (`float`, `float`) `tuple`, optional
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
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

        axes_x_limits : `float` or (`float`, `float`) or ``None``, optional
            The limits of the x axis. If `float`, then it sets padding on the
            right and left of the PointCloud as a percentage of the PointCloud's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the PointCloud as a percentage of the PointCloud's
            height. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_x_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the x axis.
        axes_y_ticks : `list` or `tuple` or ``None``, optional
            The ticks of the y axis.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            If both ``with_labels`` and ``without_labels`` are passed.
        ValueError
            If the landmark manager doesn't contain the provided group label.
        """
        if not self.has_landmarks:
            raise ValueError('PointCloud does not have landmarks attached, '
                             'unable to view landmarks.')
        self_view = self.view(figure_id=figure_id, new_figure=new_figure,
                              image_view=image_view, figure_size=figure_size)
        landmark_view = self.landmarks[group].view(
            with_labels=with_labels, without_labels=without_labels,
            figure_id=self_view.figure_id, new_figure=False,
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

        return landmark_view

    def _view_3d(self, figure_id=None, new_figure=False):
        r"""
        Visualization of the PointCloud in 3D.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.

        Returns
        -------
        viewer : PointCloudViewer3d
            The Menpo3D viewer object.
        """
        try:
            from menpo3d.visualize import PointCloudViewer3d
            return PointCloudViewer3d(figure_id, new_figure,
                                      self.points).render()
        except ImportError:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError()

    def _view_landmarks_3d(self, figure_id=None, new_figure=False,
                           group=None):
        r"""
        Visualization of the PointCloud landmarks in 3D.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        group : `str`
            The landmark group to visualize. If ``None`` is passed, the first
            and only landmark group on the object will be visualized.

        Returns
        -------
        viewer : LandmarkViewer3d
            The Menpo3D viewer object.
        """
        try:
            from menpo3d.visualize import LandmarkViewer3d
            self_renderer = self.view(figure_id=figure_id,
                                      new_figure=new_figure)
            return LandmarkViewer3d(self_renderer.figure, False,  self,
                                    self.landmarks[group]).render()
        except ImportError:
            from menpo.visualize import Menpo3dMissingError
            raise Menpo3dMissingError()

    def view_widget(self, browser_style='buttons', figure_size=(10, 8),
                    style='coloured'):
        r"""
        Visualization of the PointCloud using an interactive widget.

        Parameters
        ----------
        browser_style : {``'buttons'``, ``'slider'``}, optional
            It defines whether the selector of the objects will have the form of
            plus/minus buttons or a slider.
        figure_size : (`int`, `int`), optional
            The initial size of the rendered figure.
        style : {``'coloured'``, ``'minimal'``}, optional
            If ``'coloured'``, then the style of the widget will be coloured. If
            ``minimal``, then the style is simple using black and white colours.
        """
        try:
            from menpowidgets import visualize_pointclouds

            visualize_pointclouds(self, figure_size=figure_size, style=style,
                                  browser_style=browser_style)
        except ImportError:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def _transform_self_inplace(self, transform):
        self.points = transform(self.points)
        return self

    def distance_to(self, pointcloud, **kwargs):
        r"""
        Returns a distance matrix between this PointCloud and another.
        By default the Euclidean distance is calculated - see
        `scipy.spatial.distance.cdist` for valid kwargs to change the metric
        and other properties.

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The second pointcloud to compute distances between. This must be
            of the same dimension as this PointCloud.

        Returns
        -------
        distance_matrix: ``(n_points, n_points)`` `ndarray`
            The symmetric pairwise distance matrix between the two PointClouds
            s.t. ``distance_matrix[i, j]`` is the distance between the i'th
            point of this PointCloud and the j'th point of the input
            PointCloud.
        """
        if self.n_dims != pointcloud.n_dims:
            raise ValueError("The two PointClouds must be of the same "
                             "dimensionality.")
        return cdist(self.points, pointcloud.points, **kwargs)

    def norm(self, **kwargs):
        r"""
        Returns the norm of this PointCloud. This is a translation and
        rotation invariant measure of the point cloud's intrinsic size - in
        other words, it is always taken around the point cloud's centre.

        By default, the Frobenius norm is taken, but this can be changed by
        setting kwargs - see ``numpy.linalg.norm`` for valid options.

        Returns
        -------
        norm : `float`
            The norm of this :map:`PointCloud`
        """
        return np.linalg.norm(self.points - self.centre(), **kwargs)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the PointCloud. This is then broadcast across the dimensions
        of the PointCloud and returns a new PointCloud containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointcloud : :map:`PointCloud`
            A new pointcloud that has been masked.

        Raises
        ------
        ValueError
            Mask must have same number of points as pointcloud.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this PointCloud.')
        pc = self.copy()
        pc.points = pc.points[mask, :]
        return pc
