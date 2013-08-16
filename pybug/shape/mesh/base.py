from pybug.exceptions import DimensionalityError
from pybug.shape import PointCloud
from pybug.shape.mesh.exceptions import TriFieldError
from pybug.visualize import TriMeshViewer
from pybug.shape.mesh.normals import compute_normals


class TriMesh(PointCloud):
    """
    A pointcloud with a connectivity defined by a triangle list. These are
    designed to be explicitly 2D or 3D.
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
    def vertex_normals(self):
        """
        Compute the per-vertex normals from the current set of points and
        triangle list. Only valid for 3D dimensional meshes.
        :return: The normal at each point
        :rtype: ndarray [self.n_points, 3]
        :raises: DimensionalityError if mesh is not 3D
        """
        if self.n_dims != 3:
            raise DimensionalityError("Normals are only valid for 3D meshes")
        return compute_normals(self.points, self.trilist)[0]

    @property
    def face_normals(self):
        """
        Compute the face normals from the current set of points and
        triangle list
        :return: The normal per triangle
        :rtype: ndarray [self.n_points, 3]
        :raises: DimensionalityError if mesh is not 3D
        """
        if self.n_dims != 3:
            raise DimensionalityError("Normals are only valid for 3D meshes")
        return compute_normals(self.points, self.trilist)[1]

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

    def view(self, **kwargs):
        """
        Visualize the TriMesh.
        """
        return TriMeshViewer(self.points, self.trilist).view(**kwargs)

    # TODO: This function is totally broken at the moment
    # def new_trimesh(self, pointmask=None, astype='self'):
    #     """ Builds a new trimesh from this one.
    #     keep. Transfers across all fields, rebuilds a suitable trilist, and
    #     handles landmark and metapoint translation (or will do, still TODO!)
    #     By default will return a mesh of type(self) (i.e. FastTriMeshes will
    #     produce FastTriMeshes) but this can be overridden using the kwarg
    #     `astype`.
    #     kwargs: pointmask: a boolean mask of points that we wish to keep
    #     """
    #     #TODO this is broken due to Landmark Manager changes. Fix after new
    #     # LM manager is finished.
    #     orig_point_index = np.arange(self.n_points)
    #     if pointmask is not None:
    #         kept_points_orig_index = orig_point_index[pointmask]
    #     else:
    #         kept_points_orig_index = orig_point_index
    #     trilist_mask = np.in1d(self.trilist, kept_points_orig_index).reshape(
    #         self.trilist.shape)
    #     # remove any triangle missing any number of points
    #     tris_mask = np.all(trilist_mask, axis=1)
    #     kept_tris_orig_index = self.trilist[tris_mask]
    #     # some additional points will have to be removed as they no longer
    #     # form part of a triangle
    #     kept_points_orig_index = np.unique(kept_tris_orig_index)
    #     # the new points are easy to get
    #     new_points = self.points[kept_points_orig_index]
    #     # now we need to transfer the trilist over. First we make a new
    #     # point index
    #     kept_points_new_index = np.arange(kept_points_orig_index.shape[0])
    #     # now we build a mapping from the orig point index to the new
    #     pi_map = np.zeros(self.n_points) # point_index_mapping
    #     pi_map[kept_points_orig_index] = kept_points_new_index
    #     # trivial to now pull out the new trilist
    #     new_trilist = pi_map[kept_tris_orig_index].astype(np.uint32)
    #     if astype == 'self':
    #         trimeshcls = type(self)
    #     elif issubclass(astype, TriMesh):
    #         trimeshcls = astype
    #     else:
    #         raise Exception('The mesh type ' + str(astype) + ' is not '
    #                                                          'understood '
    #                                                          '(needs to be an'
    #                                                          ' instance of '
    #                                                          'TriMesh)')
    #     newtrimesh = trimeshcls(new_points, new_trilist)
    #     # now we just map over point fields and trifields respectively
    #     # (note that as tcoords are simply fields, this will inherently map
    #     # over our textures too)
    #     for name, field in self.pointfields.iteritems():
    #         newtrimesh.add_pointfield(name, field[kept_points_orig_index])
    #     for name, field in self.trifields.iteritems():
    #         newtrimesh.add_trifield(name, field[tris_mask])
    #     newtrimesh.texture = self.texture
    #     # TODO transfer metapoints and points.
    #     # also, convert reference landmarks to meta landmarks if their point is
    #     # removed.
    #     # TODO make this more solid - don't want to directly touch the all
    #     # landmarks
    #     # for lm in self.landmarks.reference_landmarks():
    #     #     old_index = lm.index
    #     #     if np.all(np.in1d(old_index, kept_points_orig_index)):
    #     #         # referenced point still exists, in the new mesh. add it!
    #     #         new_index = pi_map[old_index]
    #     #         newlm = ReferenceLandmark(newtrimesh, new_index,
    #     #                                   lm.label,
    #     #                                   lm.label_index)
    #     #         newtrimesh.landmarks.all_landmarks.append(newlm)
    #     #     else:
    #     #         print 'the point for landmark: ' + str(
    #     #             lm.numbered_label) + ' no longer will exist.'
    #     return newtrimesh
    #     #new_landmarks = self.landmarks.copy()
    #     #for feature in new_landmarks:
    #     #    new_landmarks[feature] = list(pi_map[new_landmarks[feature]])
