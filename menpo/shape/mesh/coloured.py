import numpy as np

from menpo.rasterize import Rasterizable
from menpo.rasterize.base import ColourRasterInfo
from menpo.visualize.base import ColouredTriMeshViewer3d

from .base import TriMesh


class ColouredTriMesh(TriMesh, Rasterizable):
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

    def copy(self):
        r"""
        An efficient copy of this ColouredTriMesh.

        Only landmarks and points will be transferred. For a full copy consider
        using `deepcopy()`.

        Returns
        -------
        colouredtrimesh: :map:`ColouredTriMesh`
            A ColouredTriMesh with the same points, trilist, colours and
            landmarks as this one.
        """
        new_ctm = ColouredTriMesh(self.points, colours=self.colours,
                                  trilist=self.trilist, copy=True)
        new_ctm.landmarks = self.landmarks
        return new_ctm

    def _view(self, figure_id=None, new_figure=False, coloured=True, **kwargs):
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
                return ColouredTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.colours).render(**kwargs)
            else:
                raise ValueError("Only viewing of 3D coloured meshes "
                                 "is currently supported.")
        else:
            return super(ColouredTriMesh, self)._view(figure_id=figure_id,
                                                      new_figure=new_figure,
                                                      **kwargs)

    @property
    def _rasterize_type_texture(self):
        return False

    def _rasterize_generate_color_mesh(self):
        return ColourRasterInfo(self.points, self.trilist, self.colours)
