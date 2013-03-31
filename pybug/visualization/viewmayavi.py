from mayavi import mlab
from tvtk.api import tvtk
from tvtk.tools import ivtk
import numpy as np
import view3d

class MayaviPointCloudViewer3d(view3d.PointCloudViewer3d):

    def __init__(self, points):
        view3d.PointCloudViewer3d.__init__(self, points)


    def view(self):
        self.currentfig = mlab.points3d(
                self.points[:,0], self.points[:,1], self.points[:,2])
        return self

class MayaviTriMeshViewer3d(view3d.TriMeshViewer3d):

    def __init__(self, points, trilist, **kwargs):
        view3d.TriMeshViewer3d.__init__(self, points, trilist, **kwargs)

    def view(self):
        self.currentfig = mlab.triangular_mesh(self.points[:,0], 
                self.points[:,1], self.points[:,2], self.trilist, 
                color=(0.5,0.5,0.5))
        return self

class MayaviTexturedTriMeshViewer3d(view3d.TexturedTriMeshViewer3d):

    def __init__(self, points, trilist, texture, **kwargs):
        view3d.TexturedTriMeshViewer3d.__init__(self, points, 
                trilist, texture, **kwargs)

    def view(self):
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.t_coords = self.tcoords_per_point
        mapper = tvtk.PolyDataMapper(input=pd)
        actor = tvtk.Actor(mapper=mapper)
        #get the texture as a np arrage and arrange it for inclusion 
        #with a tvtk ImageData class
        np_texture = np.array(self.texture)
        image_data = np.flipud(np_texture).flatten().reshape(
                [-1,3]).astype(np.uint8)
        image = tvtk.ImageData()
        image.point_data.scalars = image_data
        image.dimensions = np_texture.shape[1], np_texture.shape[0], 1
        texture = tvtk.Texture(input=image)
        actor.texture = texture
        v = ivtk.IVTK(size=(700,700))
        v.open()
        v.scene.add_actors(actor)
        currentfig = v.scene
        return self

