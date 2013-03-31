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

    def __init__(self, points, trilist, **kwargs):
        Viewer3d.__init__(self, points)
        self.trilist = trilist
        self.color_per_tri = kwargs.get('color_per_tri')
        self.color_per_point = kwargs.get('color_per_point')

class TexturedTriMeshViewer3d(TriMeshViewer3d):

    def __init__(self, points, trilist, texture, **kwargs):
        TriMeshViewer3d.__init__(self, points, trilist)
        self.texture = texture
        self.tcoords_per_tri = kwargs.get('tcoords_per_tri')
        self.tcoords_per_point = kwargs.get('tcoords_per_point')
        if self.tcoords_per_tri == None and self.tcoords_per_point == None:
            raise TCoordsViewerError("tcoords need to be provided per-point" +\
                    "or per-triangle")
        if self.tcoords_per_tri != None:
            # for now we don't render these well, and just convert to a tcoord
            # per point representation.
            self.tcoords_per_point = self.tc_per_tri_to_tc_per_point(
                    self.trilist, self.tcoords_per_tri)

    def tc_per_tri_to_tc_per_point(self, trilist, tcoords):
        """Generates per-point tc from per-tri tc. Obviously
         this means we loose data (some triangles will have fugly
         textures) but allows for quick and dirty visualization of textures.
        """
        u_ci, ind_of_u_ci = np.unique(trilist, return_index=True)
        return tcoords.reshape([-1,2])[ind_of_u_ci]
