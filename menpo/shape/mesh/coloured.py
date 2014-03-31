import numpy as np

from menpo.exception import DimensionalityError
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
    trilist : (M, 3) ndarray
        The triangle list for the mesh
    colours : (N, 3) ndarray
        The floating point RGB colour per vertex.

    Raises
    ------
    ValueError
        If the number of colour values does not match the number of vertices.
    """

    def __init__(self, points, trilist=None, colours=None):
        TriMesh.__init__(self, points, trilist=trilist)
        if colours is None:
            # default to grey
            colours = np.ones_like(points, dtype=np.float) * 0.5
        if points.shape[0] != colours.shape[0]:
            raise ValueError('Must provide a colour per-vertex.')
        self.colours = colours

    def _view(self, figure_id=None, new_figure=False, coloured=True, **kwargs):
        r"""
        Visualize the :class:`ColouredTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        coloured : bool, optional
            If ``True``, render the colours.

            Default: ``True``

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If ``self.n_dims != 3``.
        """
        if coloured:
            if self.n_dims == 3:
                return ColouredTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.colours).render(**kwargs)
            else:
                raise DimensionalityError("Only viewing of 3D coloured meshes"
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
