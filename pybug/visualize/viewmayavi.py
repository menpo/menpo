from mayavi import mlab
from tvtk.api import tvtk
import numpy as np
import abc
from pybug.visualize.base import Renderer


class MayaviViewer(Renderer):
    """
    Abstract class for performing visualizations using Mayavi.

    Parameters
    ----------
    figure_id : int or ``None``
        A figure id or ``None``. ``None`` implicitly creates a new figure.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        super(MayaviViewer, self).__init__(figure_id, new_figure)

    def get_figure(self):
        if self.new_figure or self.figure_id is not None:
            self.figure = mlab.figure(self.figure_id)
        else:
            self.figure = mlab.gcf()

        self.figure_id = self.figure.name

        return self.figure


class MayaviPointCloudViewer3d(MayaviViewer):
    def __init__(self, figure_id, new_figure, points):
        super(MayaviPointCloudViewer3d, self).__init__(figure_id, new_figure)
        self.points = points

    def _render(self, **kwargs):
        mlab.points3d(
            self.points[:, 0], self.points[:, 1], self.points[:, 2],
            figure=self.figure, scale_factor=1)
        return self


class MayaviLandmarkViewer3d(MayaviViewer):
    def __init__(self, figure_id, new_figure, label, landmark_dict):
        super(MayaviLandmarkViewer3d, self).__init__(figure_id, new_figure)
        self.label = label
        self.landmark_dict = landmark_dict

    def _render(self, **kwargs):
        # disabling the rendering greatly speeds up this for loop
        self.figure.scene.disable_render = True
        positions = []
        for label, pcloud in self.landmark_dict.iteritems():
            for i, p in enumerate(pcloud.points):
                positions.append(p)
                l = '%s_%d' % (label, i)
                # TODO: This is due to a bug in mayavi that won't allow rendering text to an empty figure
                mlab.points3d(p[0], p[1], p[2])
                mlab.text3d(p[0], p[1], p[2], l, figure=self.figure)
        positions = np.array(positions)
        os = np.zeros_like(positions)
        mlab.quiver3d(positions[:, 0], positions[:, 1], positions[:, 2],
                      os[:, 0], os[:, 1], os[:, 2], figure=self.figure)
        self.figure.scene.disable_render = False

        return self


class MayaviTriMeshViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, trilist):
        super(MayaviTriMeshViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render(self, **kwargs):
        mlab.triangular_mesh(self.points[:, 0],
                             self.points[:, 1],
                             self.points[:, 2],
                             self.trilist,
                             color=(0.5, 0.5, 0.5),
                             figure=self.figure)
        return self


class MayaviTexturedTriMeshViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points,
                 trilist, texture, tcoords_per_point):
        super(MayaviTexturedTriMeshViewer3d, self).__init__(figure_id,
                                                            new_figure)
        self.points = points
        self.trilist = trilist
        self.texture = texture
        self.tcoords_per_point = tcoords_per_point

    def _render(self, **kwargs):
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
        self.figure.scene.add_actors(actor)

        return self

