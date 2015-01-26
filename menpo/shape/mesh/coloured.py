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
    trilist : ``(M, 3)`` `ndarray` or `None`, optional
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

        if points.shape[0] != colours.shape[0]:
            raise ValueError('Must provide a colour per-vertex.')
        self.colours = colours_handle

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
        Visualize the :class:`ColouredTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        coloured : bool, optional
            If `True`, render the colours.

            Default: `True`

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If `self.n_dims != 3`.
        """
        if coloured:
            try:
                from menpo3d.visualize import ColouredTriMeshViewer3d
                return ColouredTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.colours).render(**kwargs)
            except ImportError:
                from menpo.visualize import Menpo3dErrorMessage
                raise ImportError(Menpo3dErrorMessage)
        else:
            return super(ColouredTriMesh, self).view(figure_id=figure_id,
                                                     new_figure=new_figure,
                                                     **kwargs)

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour='r', line_style='-',
                 line_width=1., render_markers=True, marker_style='o',
                 marker_size=20, marker_face_colour='k', marker_edge_colour='k',
                 marker_edge_width=1., render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, figure_size=None,
                 label=None):
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
            marker_edge_width=marker_edge_width, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            figure_size=figure_size, label=label)
