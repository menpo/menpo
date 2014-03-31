from cpptrimesh import CppTriMesh
from menpo.shape.mesh.base import TriMesh
from menpo.shape.mesh.coloured import ColouredTriMesh
from menpo.shape.pointcloud import PointCloud


class FastTriMesh(TriMesh, CppTriMesh):
    """A TriMesh with an underlying C++ data structure, allowing for efficient
    iterations around mesh vertices and triangles. Includes fast calculations
    of the surface divergence, gradient and laplacian.
    """
    #TODO this should probably be made part of Graph with some adjustments.

    def __init__(self, points, trilist):
        CppTriMesh.__init__(self, points, trilist)
        TriMesh.__init__(self, points, trilist)


class PolyMesh(PointCloud):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    polyhedrons with vertices indexed from points. This is largely a stub that
    can be expanded later on if we need arbitrary polymeshes.
    """

    def __init__(self, points, polylist):
        PointCloud.__init__(self, points)
        self.polylist = polylist

    @property
    def n_polys(self):
        return len(self.polylist)


from .textured import TexturedTriMesh
from .coloured import ColouredTriMesh
