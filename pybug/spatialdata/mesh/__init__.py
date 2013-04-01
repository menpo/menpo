import numpy as np
from .. import PointCloud3d, FieldError
from cpptrianglemesh import CppTriangleMesh
from pybug.visualization import TriMeshViewer3d, TexturedTriMeshViewer3d


class TriFieldError(FieldError):
    pass


class TriMesh(PointCloud3d):
    """A peicewise planar 3D manifold composed from triangles with vertices 
    indexed from points.
    """
    def __init__(self, points, trilist):
        PointCloud3d.__init__(self, points)
        self.trilist = trilist
        self.trifields = {}
        self.texture = None

    def attach_texture(self, texture, tcoords, tcoords_trilist=None):
        self.texture = texture
        if tcoords_trilist != None:
            # looks like we have tcoords that are referenced into
            # by a trilist in the same way points are. As it becomes messy to
            # maintain different texturing options, we just turn this indexing 
            # scheme into (repeated)  values stored explicitly as a trifield.
            self.add_trifield('tcoords', tcoords[tcoords_trilist])
        elif tcoords.shape == (self.n_points, 2):
            # tcoords are just per vertex
            self.add_pointfield('tcoords', tcoords)
        elif tcoords.shape == (self.n_tris, 3, 2):
            # explictly given per triangle evertex
            self.add_trifield('tcoords', tcoords)
        else:
            raise TextureError("Don't understand how to deal with these tcoords.")

    @property
    def n_tris(self):
        return len(self.trilist)

    def add_trifield(self, name, field):
        if field.shape[0] != self.n_tris:
            raise TriFieldError("Trying to add a field with " +
                    `field.shape[0]` + " values (need one field value per " +
                    "tri => " + `self.n_tris` + " values required")
        else:
            self.trifields[name] = field

    def view(self, textured=True):
        if textured and self.texture:
            viewer = TexturedTriMeshViewer3d(
                    self.points, self.trilist, self.texture,
                    tcoords_per_tri=self.trifields.get('tcoords'), 
                    tcoords_per_point=self.pointfields.get('tcoords'))
        else:
            viewer = TriMeshViewer3d(self.points, self.trilist, 
                    color_per_tri=self.trifields.get('color'), 
                    color_per_point=self.pointfields.get('color'))

        return viewer.view()


class FastTriMesh(TriMesh, CppTriangleMesh):

    def __init__(self, points, trilist):
        CppTriangleMesh.__init__(self, points, trilist)
        TriMesh.__init__(self, points, trilist)


class PolyMesh(PointCloud3d):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    polyhedrons with vertices indexed from points. This is largely a stub that
    can be expanded later on if we need arbitrary polymeshes.
    """
    def __init__(self, points, polylist):
        PointCloud3d.__init__(self, points)
        self.polylist = polylist

    @property
    def n_polys(self):
        return len(self.polylist)



