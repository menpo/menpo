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

    def trimesh_from_pointmask(self, pointmask):
        orig_point_index = np.arange(self.n_points)
        kept_points_orig_index = orig_point_index[pointmask]
        trilist_mask = np.in1d(self.trilist, kept_points_orig_index).reshape(
                self.trilist.shape)
        # remove any triangle missing any number of points
        tris_mask = np.all(trilist_mask, axis = 1)
        kept_tris_orig_index = self.trilist[tris_mask]
        # some additional points will have to be removed as they no longer
        # form part of a triangle
        kept_points_orig_index = np.unique(kept_tris_orig_index)
        # the new points are easy to get
        new_points = self.points[kept_points_orig_index]
        # now we need to transfer the trilist over. First we make a new 
        # point index
        kept_points_new_index = np.arange(kept_points_orig_index.shape[0])
        # now we build a mapping from the orig point index to the new
        pi_map = np.zeros(self.n_points) # point_index_mapping
        pi_map[kept_points_orig_index] = kept_points_new_index
        # trivial to now pull out the new trilist
        new_trilist = pi_map[kept_tris_orig_index].astype(np.uint32)
        newtrimesh = TriMesh(new_points, new_trilist)
        # now we just map over point fields and trifields respectively
        # (note that as tcoords are simply fields, this will inherently map
        # over our textures too)
        for name, field in self.pointfields.iteritems():
            newtrimesh.add_pointfield(name, field[kept_points_orig_index])
        for name, field in self.trifields.iteritems():
            newtrimesh.add_trifield(name, field[tris_mask])
        newtrimesh.texture = self.texture
        # TODO transfer metapoints and landmarks
        return newtrimesh
        #new_landmarks = self.landmarks.copy()
        #for feature in new_landmarks:
        #    new_landmarks[feature] = list(pi_map[new_landmarks[feature]])


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



