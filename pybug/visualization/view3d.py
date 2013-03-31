import numpy as np

class Viewer3dError(Exception):
    pass

class TCoordsViewerError(Viewer3dError):
    pass

class Viewer(object):
    """Abstract class for performaing visualizations. Framework specific 
    implimentations of these classes are made in order to seperate implimentation
    cleanly from the rest of the code.
    """

    def __init__(self):
        self.currentfig = None

class Viewer3d(Viewer):

    def __init__(self, points):
        Viewer.__init__(self)
        dim = points.shape[1]
        if dim != 3:
            raise Viewer3dError("Trying to view " + str(dim) +\
                    "data with a 3DViewer")
        self.points = points


class PointCloudViewer3d(Viewer3d):

    def __init__(self, points):
        Viewer3d.__init__(self, points)


class TriMeshViewer3d(Viewer3d):

    def __init__(self, points, trilist):
        Viewer3d.__init__(self, points)
        self.trilist = trilist

class TexturedTriMeshViewer3d(TriMeshViewer3d):

    def __init__(self, points, trilist, tcoords, texture, texture_trilist=None):
        TriMeshViewer3d.__init__(self, points)
        self._raw_tcoords, self.texture = tcoords, texture
        n_tcoords = tcoords.shape[0]
        if n_t_coords == points.shape[0]:
            self.tcoords_per_point = self._raw_tcoords
            # texture coords are per point
        elif n_t_coords == trilist.shape[0]:
            # texture coords are per triangle
            self.tcoords_per_tri = self._raw_tcoords
            self.texture_trilist = texture_trilist
            self.tcoords_per_point = self.tc_per_tri_to_tc_per_point(
                    self.trilist, self.texture_trilist, self.tcoords_per_tri)
        else:
            raise TCoordsViewerError("tcoords need to be one per-point" +\
                    "or per-triangle")

    def tc_per_tri_to_tc_per_point(self, trilist, texture_trilist, tc):
        """Generates per-point tc from per-tri tc. Obviously
         this means we loose data (some triangles will have fugly
         textures) but allows for quick and dirty visualization of textures.
        """
        u_ci, ind_of_u_ci = np.unique(trilist, return_index=True)
        # grab these positions from the texture_coord_index to find an instance
        # of a tc at each vertex
        per_vertex_tci = texture_trilist.flatten()[ind_of_u_ci]
        return tc[per_vertex_tci]
