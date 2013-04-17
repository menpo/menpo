import numpy as np
from pybug import spatialdata
from cpptrianglemesh import CppTriangleMesh
from pybug.visualization import TriMeshViewer3d, TexturedTriMeshViewer3d


class TriFieldError(spatialdata.FieldError):
    pass


class TextureError(Exception):
    pass


class TriMesh(spatialdata.PointCloud3d):
    """A piecewise planar 3D manifold composed from triangles with vertices
    indexed from points.
    """

    def __init__(self, points, trilist):
        spatialdata.PointCloud3d.__init__(self, points)
        self.trilist = trilist
        self.trifields = {}
        self.texture = None

    def __str__(self):
        message = spatialdata.PointCloud.__str__(self)
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

    def attach_texture(self, texture, tcoords, tcoords_trilist=None):
        """Attaches a trifield or pointfield called 'tcoords' depending
        on whether the tcoords given are per vertex or per triangle.
        kwargs:
           tcoords_trilist: a texture specific trilist used to index into
           the tcoords. In this case tcoords will be converted to a trifield
           removing the dependency on the texture specific trilist.
           This comes at a memory cost (there will be many repeated tcoords in
           the constructed trifield), but allows for a consistent processing
           of texture coords as just another field instance.
        """
        self.texture = texture
        if tcoords_trilist is not None:
            # looks like we have tcoords that are referenced into
            # by a trilist in the same way points are. As it becomes messy to
            # maintain different texturing options, we just turn this indexing
            # scheme into (repeated) values stored explicitly as a trifield.
            self.add_trifield('tcoords', tcoords[tcoords_trilist])
        elif tcoords.shape == (self.n_points, 2):
            # tcoords are just per vertex
            self.add_pointfield('tcoords', tcoords)
        elif tcoords.shape == (self.n_tris, 3, 2):
            # explicitly given per triangle vertex
            self.add_trifield('tcoords', tcoords)
        else:
            raise TextureError(
                "Don't understand how to deal with these tcoords.")

    @property
    def n_tris(self):
        return len(self.trilist)

    def add_trifield(self, name, field):
        if field.shape[0] != self.n_tris:
            raise TriFieldError("Trying to add a field with " +
                                str(field.shape[0]) + " values (need one "
                                "field value per tri => " +
                                str(self.n_tris) + ")")
        else:
            self.trifields[name] = field

    def view(self, textured=True):
        """ Visualize the TriMesh. By default, if the mesh has a texture a
        textured view will be provided. This can be overridden using the
        boolean kwarg `textured`
        """
        if textured and self.texture:
            viewer = TexturedTriMeshViewer3d(
                self.points, self.trilist, self.texture,
                tcoords_per_tri=self.trifields.get('tcoords'),
                tcoords_per_point=self.pointfields.get('tcoords'))
        else:
            viewer = TriMeshViewer3d(self.points, self.trilist,
                                     color_per_tri=self.trifields.get('color'),
                                     color_per_point=self.pointfields.get(
                                         'color'))
        return viewer.view()

    def new_trimesh(self, pointmask=None, astype='self'):
        """ Builds a new trimesh from this one.
        keep. Transfers across all fields, rebuilds a suitable trilist, and
        handles landmark and metapoint translation (or will do, still TODO!)
        By default will return a mesh of type(self) (i.e. FastTriMeshes will
        produce FastTriMeshes) but this can be overridden using the kwarg
        `astype`.
        kwargs: pointmask: a boolean mask of points that we wish to keep
        """
        orig_point_index = np.arange(self.n_points)
        if pointmask is not None:
            kept_points_orig_index = orig_point_index[pointmask]
        else:
            kept_points_orig_index = orig_point_index
        trilist_mask = np.in1d(self.trilist, kept_points_orig_index).reshape(
            self.trilist.shape)
        # remove any triangle missing any number of points
        tris_mask = np.all(trilist_mask, axis=1)
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
        if astype == 'self':
            trimeshcls = type(self)
        elif issubclass(astype, TriMesh):
            trimeshcls = astype
        else:
            raise Exception('The mesh type ' + str(astype) + ' is not '
                            'understood (needs to be an instance of '
                            'TriMesh)')
        newtrimesh = trimeshcls(new_points, new_trilist)
        # now we just map over point fields and trifields respectively
        # (note that as tcoords are simply fields, this will inherently map
        # over our textures too)
        for name, field in self.pointfields.iteritems():
            newtrimesh.add_pointfield(name, field[kept_points_orig_index])
        for name, field in self.trifields.iteritems():
            newtrimesh.add_trifield(name, field[tris_mask])
        newtrimesh.texture = self.texture
        # TODO transfer metapoints and points.
        # also, convert reference landmarks to meta landmarks if their point is
        # removed.
        # TODO make this more solid - don't want to directly touch the all
        # landmarks
        for lm in self.landmarks.reference_landmarks():
            old_index = lm.index
            if np.all(np.in1d(old_index, kept_points_orig_index)):
                # referenced point still exists, in the new mesh. add it!
                new_index = pi_map[old_index]
                newlm = spatialdata.ReferenceLandmark(newtrimesh, new_index,
                                                      lm.label,
                                                      lm.label_index)
                newtrimesh.landmarks.all_landmarks.append(newlm)
            else:
                print 'the point for landmark: ' + str(
                    lm.numbered_label) + ' no longer will exist.'
        return newtrimesh
        #new_landmarks = self.landmarks.copy()
        #for feature in new_landmarks:
        #    new_landmarks[feature] = list(pi_map[new_landmarks[feature]])


class FastTriMesh(TriMesh, CppTriangleMesh):
    """A TriMesh with an underlying C++ data structure, allowing for efficient
    iterations around mesh vertices and triangles. Includes fast calculations
    of the surface divergence, gradient and laplacian.
    """

    def __init__(self, points, trilist):
        CppTriangleMesh.__init__(self, points, trilist)
        TriMesh.__init__(self, points, trilist)


class PolyMesh(spatialdata.PointCloud3d):
    """A 3D shape which has a notion of a manifold built from piecewise planar
    polyhedrons with vertices indexed from points. This is largely a stub that
    can be expanded later on if we need arbitrary polymeshes.
    """

    def __init__(self, points, polylist):
        spatialdata.PointCloud3d.__init__(self, points)
        self.polylist = polylist

    @property
    def n_polys(self):
        return len(self.polylist)


class TriMeshShapeClass(spatialdata.ShapeClass):
    """A shape class that only contains TriMesh instances
    """

    def __init__(self, trimeshiter):
        spatialdata.ShapeClass.__init__(self, trimeshiter)
        if not all(isinstance(x, TriMesh) for x in self.data):
            raise spatialdata.ShapeClassError("Trying to build a trimesh shape"
                                              " class with non-trimesh "
                                              "elements")