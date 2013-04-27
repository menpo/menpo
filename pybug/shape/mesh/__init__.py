import numpy as np
from cpptrianglemesh import CppTriangleMesh
from pybug.shape.exceptions import FieldError
from pybug.shape.landmarks import ReferenceLandmark
from pybug.shape.pointcloud import PointCloud
from pybug.visualization import TriMeshViewer3d


class TriFieldError(FieldError):
    pass


class TextureError(Exception):
    pass


class TriMesh(PointCloud):
    """A piecewise planar 3D manifold composed from triangles with vertices
    indexed from points.
    """

    def __init__(self, points, trilist):
        #TODO delauney triangulate if no trilist added
        PointCloud.__init__(self, points)
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
        #TODO this is broken due to Landmark Manager changes. Fix after new
        # LM manager is finished.
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
                                                             'understood '
                                                             '(needs to be an'
                                                             ' instance of '
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
                newlm = ReferenceLandmark(newtrimesh, new_index,
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
    #TODO this should probably be made part of Graph with some adjustments.

    def __init__(self, points, trilist):
        CppTriangleMesh.__init__(self, points, trilist)
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