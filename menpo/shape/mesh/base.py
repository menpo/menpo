import numpy as np
from scipy.spatial import Delaunay

from menpo.shape import PointCloud
from menpo.shape.mesh.normals import compute_normals
from menpo.visualize import TriMeshViewer
from menpo.exception import DimensionalityError


class TriMesh(PointCloud):
    r"""
    A pointcloud with a connectivity defined by a triangle list. These are
    designed to be explicitly 2D or 3D.

    Parameters
    ----------
    points : (N, D) ndarray
        The set coordinates for the mesh.
    trilist : (M, 3) ndarray, optional
        The triangle list. If None is provided, a Delaunay triangulation of
        the points will be used instead.

        Default: None
    """

    def __init__(self, points, trilist=None):
        #TODO add inheritance from Graph once implemented
        super(TriMesh, self).__init__(points)
        if trilist is None:
            trilist = Delaunay(points).simplices
        self.trilist = np.array(trilist, copy=True, order='C')

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

    def from_vector(self, flattened):
        r"""
        Builds a new :class:`TriMesh` given then ``flattened`` vector.
        This allows rebuilding pointclouds with the correct number of
        dimensions from a vector. Note that the trilist will be drawn from
        self.

        Parameters
        ----------
        flattened : (N,) ndarray
            Vector representing a set of points.

        Returns
        --------
        trimesh : :class:`TriMesh`
            A new trimesh created from the vector with self's trilist.
        """
        return TriMesh(flattened.reshape([-1, self.n_dims]), self.trilist)

    @property
    def vertex_normals(self):
        r"""
        Normal at each point.

        :type: (``n_points``, 3) ndarray

        Compute the per-vertex normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Raises
        ------
        DimensionalityError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise DimensionalityError("Normals are only valid for 3D meshes")
        return compute_normals(self.points, self.trilist)[0]

    @property
    def face_normals(self):
        r"""
        Normal at each face.

        :type: (``n_tris``, 3) ndarray

        Compute the face normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.

        Raises
        ------
        DimensionalityError
            If mesh is not 3D
        """
        if self.n_dims != 3:
            raise DimensionalityError("Normals are only valid for 3D meshes")
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
            If ``not self.n_dims in [2, 3]``.
        """
        return TriMeshViewer(figure_id, new_figure,
                             self.points, self.trilist).render(**kwargs)
