from pybug.shape import PointCloud
from pybug.shape.mesh.exceptions import TriFieldError
from pybug.visualize import TriMeshViewer


class TriMesh(PointCloud):
    r"""
    A pointcloud with a connectivity defined by a triangle list. These are
    designed to be explicitly 2D or 3D.

    Parameters
    ----------
    points : (N, D) ndarray
        The set coordinates for the mesh.
    trilist : (M, 3) ndarray
        The triangle list.
    """

    def __init__(self, points, trilist):
        #TODO Delaunay triangulate if no trilist added
        #TODO add inheritance from Graph once implemented
        super(TriMesh, self).__init__(points)
        self.trilist = trilist
        self.trifields = {}

    def __str__(self):
        message = PointCloud.__str__(self)
        if len(self.trifields) != 0:
            message += '\n  trifields:'
            for k, v in self.trifields.iteritems():
                try:
                    field_dim = v.shape[1]
                except IndexError:
                    field_dim = 1
                message += '\n    ' + str(k) + '(' + str(field_dim) + 'D)'
        message += '\nn_tris: ' + str(self.n_tris)
        return message

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: int
        """
        return len(self.trilist)

    def add_trifield(self, name, field):
        if field.shape[0] != self.n_tris:
            raise TriFieldError("Trying to add a field with " +
                                str(field.shape[0]) + " values (need one "
                                "field value per tri => " +
                                str(self.n_tris) + ")")
        else:
            self.trifields[name] = field

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        """
        Visualize the TriMesh.

        Parameters
        ----------
        kwargs : dict
            Passed through to the viewer.

        Returns
        -------
        viewer : :class:`pybug.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If ``not self.n_dims in [2, 3]``.
        """
        return TriMeshViewer(figure_id, new_figure,
                             self.points, self.trilist).render(**kwargs)
