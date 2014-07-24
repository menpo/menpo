import numpy as np
from warnings import warn
from scipy.spatial import Delaunay

from menpo.shape import PointCloud
from menpo.rasterize import Rasterizable, ColourRasterInfo
from menpo.visualize import TriMeshViewer

from .normals import compute_normals


class TriMesh(PointCloud, Rasterizable):
    r"""
    A pointcloud with a connectivity defined by a triangle list. These are
    designed to be explicitly 2D or 3D.

    Parameters
    ----------
    points : (N, D) ndarray
        The set coordinates for the mesh.
    trilist : (M, 3) ndarray, optional
        The triangle list. If `None`, a Delaunay triangulation of
        the points will be used instead.

        Default: None
    copy: bool, optional
        If `False`, the points will not be copied on assignment.
        Any trilist will also not be copied.
        In general this should only be used if you know what you are doing.

        Default: `False`
    """

    def __init__(self, points, trilist=None, copy=True):
        #TODO add inheritance from Graph once implemented
        super(TriMesh, self).__init__(points, copy=copy)
        if trilist is None:
            trilist = Delaunay(points).simplices
        if not copy:
            if not trilist.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                trilist = np.array(trilist, copy=True, order='C')
        else:
            trilist = np.array(trilist, copy=True, order='C')
        self.trilist = trilist

    @property
    def _rasterize_type_texture(self):
        return False

    def _rasterize_generate_color_mesh(self):
        return ColourRasterInfo(self.points, self.trilist, self.points)

    def __str__(self):
        return '{}, n_tris: {}'.format(PointCloud.__str__(self),
                                       self.n_tris)

    def tojson(self):
        r"""
        Convert this `TriMesh` to a dictionary JSON representation.

        Returns
        -------
        dictionary with 'points' and 'trilist' keys. Both are lists suitable
        for use in the by the `json` standard library package.
        """
        json_dict = PointCloud.tojson(self)
        json_dict['trilist'] = self.trilist.tolist()
        return json_dict

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the pointcloud. This is then broadcast across the dimensions
        of the pointcloud and returns a new pointcloud containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointcloud : :map:`PointCloud`
            A new pointcloud that has been masked.
        """
        tm = self.copy()
        if np.all(mask):  # Fast path for all true
            return tm
        return tm

    @property
    def vertex_normals(self):
        r"""
        Normal at each point.

        :type: (`n_points`, 3) ndarray

        Compute the per-vertex normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Raises
        ------
        DimensionalityError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_normals(self.points, self.trilist)[0]

    @property
    def face_normals(self):
        r"""
        Normal at each face.

        :type: (`n_tris`, 3) ndarray

        Compute the face normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Raises
        ------
        DimensionalityError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise ValueError("Normals are only valid for 3D meshes")
        return compute_normals(self.points, self.trilist)[1]

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: int
        """
        return len(self.trilist)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        """
        Visualize the TriMesh.

        Parameters
        ----------
        kwargs : dict
            Passed through to the viewer.

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If `not self.n_dims in [2, 3]`.
        """
        return TriMeshViewer(figure_id, new_figure,
                             self.points, self.trilist).render(**kwargs)
