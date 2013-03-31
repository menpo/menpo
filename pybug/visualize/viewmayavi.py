from mayavi import mlab
import view3d

class MayaviPointCloudViewer3d(view3d.PointCloudViewer3d):

    def __init__(self, points):
        view3d.PointCloudViewer3d.__init__(self, points)


    def view(self):
        self.currentfig = mlab.points3d(
                self.points[:,0], self.points[:,1], self.points[:,2])
        return self

class MayaviTriMeshViewer3d(Viewer3d):

    def __init__(self, points, trilist):
        Viewer3d.__init__(self, points)
        self.trilist = trilist

    def view(self):
        self.currentfig = mlab.triangular_mesh(self.points[:,0], 
                self.points[:,1], self.points[:,2], self.trilist, 
                color=(0.5,0.5,0.5))
        return self
