import numpy as np

from ..adjacency import mask_adjacency_array, reindex_adjacency_array
from .base import TriMesh


class ColouredTriMesh(TriMesh):
    r"""
    Combines a :map:`TriMesh` with a colour per vertex.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The array representing the points.
    trilist : ``(M, 3)`` `ndarray` or ``None``, optional
        The triangle list. If `None`, a Delaunay triangulation of
        the points will be used instead.
    colours : ``(N, 3)`` `ndarray`, optional
        The floating point RGB colour per vertex. If not given, grey will be
        assigned to each vertex.
    copy: `bool`, optional
        If ``False``, the points, trilist and colours will not be copied on
        assignment.
        In general this should only be used if you know what you are doing.

    Raises
    ------
    ValueError
        If the number of colour values does not match the number of vertices.
    """

    def __init__(self, points, trilist=None, colours=None, copy=True):
        TriMesh.__init__(self, points, trilist=trilist, copy=copy)
        # Handle the settings of colours, either be provided a default grey
        # set of colours, or copy the given array if necessary
        if colours is None:
            # default to grey
            colours_handle = np.ones_like(points, dtype=np.float) * 0.5
        elif not copy:
            colours_handle = colours
        else:
            colours_handle = colours.copy()

        if points.shape[0] != colours_handle.shape[0]:
            raise ValueError('Must provide a colour per-vertex.')
        self.colours = colours_handle

    @classmethod
    def init_2d_grid(cls, shape, spacing=None, colours=None):
        r"""
        Create a ColouredTriMesh that exists on a regular 2D grid. The first
        dimension is the number of rows in the grid and the second dimension
        of the shape is the number of columns. ``spacing`` optionally allows
        the definition of the distance between points (uniform over points).
        The spacing may be different for rows and columns.

        The triangulation will be right-handed and the diagonal will go from
        the top left to the bottom right of a square on the grid.

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
        colours : ``(N, 3)`` `ndarray`, optional
            The floating point RGB colour per vertex. If not given, grey will be
            assigned to each vertex.

        Returns
        -------
        trimesh : :map:`TriMesh`
            A TriMesh arranged in a grid.
        """
        pc = TriMesh.init_2d_grid(shape, spacing=spacing)
        points = pc.points
        trilist = pc.trilist
        # Ensure that the colours are copied
        if colours is not None:
            colours = colours.copy()
        return ColouredTriMesh(points, trilist=trilist, colours=colours,
                               copy=False)

    @classmethod
    def init_from_depth_image(cls, depth_image, colours=None):
        r"""
        Return a 3D textured triangular mesh from the given depth image. The
        depth image is assumed to represent height/depth values and the XY
        coordinates are assumed to unit spaced and represent image coordinates.
        This is particularly useful for visualising depth values that have been
        recovered from images.

        The optionally passed texture will be textured mapped onto the planar
        surface using the correct texture coordinates for an image of the
        same shape as ``depth_image``.

        Parameters
        ----------
        depth_image : :map:`Image` or subclass
            A single channel image that contains depth values - as commonly
            returned by RGBD cameras, for example.
        colours : ``(N, 3)`` `ndarray`, optional
            The floating point RGB colour per vertex. If not given, grey will be
            assigned to each vertex.

        Returns
        -------
        depth_cloud : ``type(cls)``
            A new 3D TriMesh with unit XY coordinates and the given depth
            values as Z coordinates. The trilist is constructed as in
            :meth:`init_2d_grid`.
        """
        from menpo.image import MaskedImage

        new_tmesh = cls.init_2d_grid(depth_image.shape, colours=colours)
        if isinstance(depth_image, MaskedImage):
            new_tmesh = new_tmesh.from_mask(depth_image.mask.as_vector())
        return cls(np.hstack([new_tmesh.points,
                              depth_image.as_vector(keep_channels=True).T]),
                   colours=new_tmesh.colours,
                   trilist=new_tmesh.trilist,
                   copy=False)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the ColouredTriMesh. This is then broadcast across the
        dimensions of the mesh and returns a new mesh containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`ColouredTriMesh`
            A new mesh that has been masked.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this '
                             'ColouredTriMesh.')

        ctm = self.copy()
        if np.all(mask):  # Fast path for all true
            return ctm
        else:
            # Recalculate the mask to remove isolated vertices
            isolated_mask = self._isolated_mask(mask)
            # Recreate the adjacency array with the updated mask
            masked_adj = mask_adjacency_array(isolated_mask, self.trilist)
            ctm.trilist = reindex_adjacency_array(masked_adj)
            ctm.points = ctm.points[isolated_mask, :]
            ctm.colours = ctm.colours[isolated_mask, :]
            return ctm

    def _view_3d(self, figure_id=None, new_figure=False, coloured=True,
                 **kwargs):
        r"""
        Visualize the :map:`ColouredTriMesh` in 3D.

        Parameters
        ----------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        coloured : `bool`, optional
            If `True`, render the colours.

        Returns
        -------
        viewer : :map:`Renderer`
            The viewer object.
        """
        if coloured:
            try:
                from menpo3d.visualize import ColouredTriMeshViewer3d
                return ColouredTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.colours).render(**kwargs)
            except ImportError:
                from menpo.visualize import Menpo3dMissingError
                raise Menpo3dMissingError()
        else:
            return super(ColouredTriMesh, self).view(figure_id=figure_id,
                                                     new_figure=new_figure,
                                                     **kwargs)

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour='r', line_style='-',
                 line_width=1., render_markers=True, marker_style='o',
                 marker_size=5, marker_face_colour='k', marker_edge_colour='k',
                 marker_edge_width=1., render_numbering=False,
                 numbers_horizontal_align='center',
                 numbers_vertical_align='bottom',
                 numbers_font_name='sans-serif', numbers_font_size=10,
                 numbers_font_style='normal', numbers_font_weight='normal',
                 numbers_font_colour='k', render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, axes_x_ticks=None,
                 axes_y_ticks=None, figure_size=(10, 8), label=None):
        r"""
        Visualization of the TriMesh in 2D. Currently, explicit coloured TriMesh
        viewing is not supported, and therefore viewing falls back to uncoloured
        2D TriMesh viewing.

        Returns
        -------
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        image_view : `bool`, optional
            If ``True`` the ColouredTriMesh will be viewed as if it is in the
            image coordinate system.
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
            The size of the markers in points.
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
            right and left of the TriMesh as a percentage of the TriMesh's
            width. If `tuple` or `list`, then it defines the axis limits. If
            ``None``, then the limits are set automatically.
        axes_y_limits : (`float`, `float`) `tuple` or ``None``, optional
            The limits of the y axis. If `float`, then it sets padding on the
            top and bottom of the TriMesh as a percentage of the TriMesh's
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

        Raises
        ------
        warning
            2D Viewing of Coloured TriMeshes is not supported, automatically
            falls back to 2D :map:`TriMesh` viewing.
        """
        import warnings
        warnings.warn(Warning('2D Viewing of Coloured TriMeshes is not '
                              'supported, falling back to TriMesh viewing.'))
        return TriMesh._view_2d(
            self, figure_id=figure_id, new_figure=new_figure,
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
            numbers_font_colour=numbers_font_colour, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            axes_x_ticks=axes_x_ticks, axes_y_ticks=axes_y_ticks,
            figure_size=figure_size, label=label)
