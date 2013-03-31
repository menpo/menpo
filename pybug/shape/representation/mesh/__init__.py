import numpy as np
from .. import PointCloud3d, FieldError
from cpptrianglemesh import CppTriangleMesh
from pybug.visualization import TriMeshViewer3d


class TriFieldError(FieldError):
    pass

class PolyMesh(PointCloud3d):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    polyhedrons with vertices indexed from points.
    """
    def __init__(self, points, polylist):
        PointCloud3d.__init__(self, points)
        self.polylist = polylist

    @property
    def n_polys(self):
        return len(self.polylist)

class TriMesh(PointCloud3d):
    """A peicewise planar 3D manifold composed from triangles with vertices 
    indexed from points.
    """
    def __init__(self, points, trilist):
        PointCloud3d.__init__(self, points)
        self.trilist = trilist
        self.trifields = {}

    @property
    def n_tris(self):
        return len(self.trilist)

    def add_trifield(self, name, field):
        if field.shape[0] != self.n_tris:
            raise TriFieldError("Trying to add a field with " +
                    `field.shape[0]` + " values (need one field value per " +
                    "tri => " + `self.n_tris` + " values required")
        else:
            self.pointfields[name] = field

    def view(self):
        viewer = TriMeshViewer3d(self.points, self.trilist)
        return viewer.view()


class FastTriMesh(TriMesh, CppTriangleMesh):

    def __init__(self, points, trilist):
        CppTriangleMesh.__init__(self, points, trilist)
        TriMesh.__init__(self, points, trilist)

