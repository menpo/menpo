from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
import view3d


class MayaviViewer(object):
    def newfigure(self):
        return mlab.figure()


class MayaviPointCloudViewer3d(view3d.PointCloudViewer3d, MayaviViewer):
    def __init__(self, points):
        view3d.PointCloudViewer3d.__init__(self, points)

    def _viewonfigure(self, figure, **kwargs):
        self.currentscene = mlab.points3d(
            self.points[:, 0], self.points[:, 1], self.points[:, 2],
            figure=figure, scale_factor=1)
        self.currentfigure = figure
        return self


class MayaviLabelViewer3d(view3d.LabelViewer3d, MayaviViewer):
    def __init__(self, points, labels, **kwargs):
        view3d.LabelViewer3d.__init__(self, points, labels, **kwargs)

    def _viewonfigure(self, figure, **kwargs):
        # disabling the rendering greatly speeds up this for loop
        figure.scene.disable_render = True
        label_pos = self.points + self.offset
        for i, label in enumerate(self.labels):
            x, y, z = tuple(label_pos[i])
            mlab.text3d(x, y, z, label, figure=figure)
        lp = label_pos
        os = -1.0 * self.offset
        mlab.quiver3d(lp[:, 0], lp[:, 1], lp[:, 2],
                      os[:, 0], os[:, 1], os[:, 2], figure=figure)
        figure.scene.disable_render = False
        self.currentfigure = figure
        return self


class MayaviTriMeshViewer3d(view3d.TriMeshViewer3d, MayaviViewer):
    def __init__(self, points, trilist, **kwargs):
        view3d.TriMeshViewer3d.__init__(self, points, trilist, **kwargs)

    def _viewonfigure(self, figure, **kwargs):
        self.currentscene = mlab.triangular_mesh(self.points[:, 0],
                                                 self.points[:, 1],
                                                 self.points[:, 2],
                                                 self.trilist,
                                                 color=(0.5, 0.5, 0.5),
                                                 figure=figure)
        self.currentfigure = figure
        return self


class MayaviTexturedTriMeshViewer3d(view3d.TexturedTriMeshViewer3d,
                                    MayaviViewer):
    def __init__(self, points, trilist, texture, **kwargs):
        view3d.TexturedTriMeshViewer3d.__init__(self, points,
                                                trilist, texture, **kwargs)

    def _viewonfigure(self, figure, **kwargs):
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.t_coords = self.tcoords_per_point
        mapper = tvtk.PolyDataMapper(input=pd)
        actor = tvtk.Actor(mapper=mapper)
        #get the texture as a np arrage and arrange it for inclusion
        #with a tvtk ImageData class
        image_data = np.flipud(self.texture.pixels).flatten().reshape(
            [-1, 3]).astype(np.uint8)
        image = tvtk.ImageData()
        image.point_data.scalars = image_data
        image.dimensions = self.texture.height, self.texture.width, 1
        texture = tvtk.Texture(input=image)
        actor.texture = texture
        #v = ivtk.IVTK(size=(700,700))
        #v.open()
        figure.scene.add_actors(actor)
        #mlab.show()
        self.currentfigure = figure
        self.currentscene = figure.scene
        return self

