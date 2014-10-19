import numpy as np

from ..adjacency import mask_adjacency_array, reindex_adjacency_array
from .base import TriMesh


class ColouredTriMesh(TriMesh):
    r"""
    Combines a :class:`menpo.shape.mesh.base.TriMesh` with a colour per vertex.

    Parameters
    ----------
    points : (N, D) ndarray
        The coordinates of the mesh.
    trilist : (M, 3) ndarray, optional
        The triangle list for the mesh. If `None`, a Delaunay triangulation
        will be performed.

        Default: `None`
    colours : (N, 3) ndarray, optional
        The floating point RGB colour per vertex. If not given, grey will be
        assigned to each vertex.

        Default: `None`
    copy: bool, optional
        If `False`, the points, trilist and colours will not be copied on
        assignment.
        In general this should only be used if you know what you are doing.

        Default: `False`

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

    def view(self, figure_id=None, new_figure=False, coloured=True, **kwargs):
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
            if self.n_dims == 3:
                try:
                    from menpo3d.visualize import ColouredTriMeshViewer3d
                    return ColouredTriMeshViewer3d(
                        figure_id, new_figure, self.points,
                        self.trilist, self.colours).render(**kwargs)
                except ImportError:
                    from menpo.visualize import Menpo3dErrorMessage
                    raise ImportError(Menpo3dErrorMessage)
            else:
                raise ValueError("Only viewing of 3D coloured meshes "
                                 "is currently supported.")
        else:
            return super(ColouredTriMesh, self).view(figure_id=figure_id,
                                                     new_figure=new_figure,
                                                     **kwargs)
