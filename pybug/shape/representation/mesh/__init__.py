import numpy as np
from .. import PointCloud
from cpptrianglemesh import CppTriangleMesh


class MeshConstructionError(Exception):
    pass

class PolyMesh(PointCloud):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    polyhedrons with vertices indexed from points.
    """
    def __init__(self, points, polylist):
        PointCloud.__init__(self, points)
        self.polylist = polylist

    @property
    def n_polys(self):
        return len(self.polylist)

class TriMesh(PointCloud):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    triangles with vertices indexed from points.
    """
    def __init__(self, points, trilist):
        PointCloud.__init__(self, points)
        self.trilist = trilist
        if self.n_dims != 3:
            raise MeshConstructionError("A Mesh shape can only be in 3 "\
                    " dimensions (attemping with " + str(self.n_dims) + ")")

    @property
    def n_tris(self):
        return len(self.polylist)


class FastTriMesh(TriMesh, CppTriangleMesh):

    def __init__(self, points, trilist):
        CppTriangleMesh.__init__(self, points, trilist)
        TriMesh.__init__(self, points, trilist)
