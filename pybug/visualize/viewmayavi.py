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
    figure_id : str or ``None``
        A figure name or ``None``. ``None`` assumes we maintain the Mayavi
        state machine and use ``mlab.gcf()``.
    new_figure : bool
        If ``True``, creates a new figure to render on.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, figure_id, new_figure):
        super(MayaviViewer, self).__init__(figure_id, new_figure)

    def get_figure(self):
        r"""
        Gets the figure specified by the combination of ``self.figure_id`` and
        ``self.new_figure``. If ``self.figure_id == None`` then ``mlab.gcf()``
        is used. ``self.figure_id`` is also set to the correct id of the figure
        if a new figure is created.

        Returns
        -------
        figure : Mayavi figure object
            The figure we will be rendering on.
        """
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


class MayaviSurfaceViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, values):
        super(MayaviSurfaceViewer3d, self).__init__(figure_id, new_figure)
        self.values = values

    def _render(self, **kwargs):
        warp_scale = kwargs.get('warp_scale', 'auto')
        mlab.surf(self.values, warp_scale=warp_scale)
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
        os[:, 2] = 1
        mlab.quiver3d(positions[:, 0], positions[:, 1], positions[:, 2],
                      os[:, 0], os[:, 1], os[:, 2], figure=self.figure)
        self.figure.scene.disable_render = False

        return self


class MayaviTriMeshViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, trilist):
        super(MayaviTriMeshViewer3d, self).__init__(figure_id, new_figure)
        self.points = points
        self.trilist = trilist

    def _render(self, normals=None, **kwargs):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals)._render(**kwargs)
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

    def _render(self, normals=None, **kwargs):
        if normals is not None:
            MayaviVectorViewer3d(self.figure_id, False,
                                 self.points, normals)._render(**kwargs)
        pd = tvtk.PolyData()
        pd.points = self.points
        pd.polys = self.trilist
        pd.point_data.t_coords = self.tcoords_per_point
        mapper = tvtk.PolyDataMapper(input=pd)
        actor = tvtk.Actor(mapper=mapper)
        # Get the pixels from our image class which are [0, 1] and scale
        # back to valid pixels. Then convert to tvtk ImageData.
        image_data = np.flipud(self.texture.pixels * 255).flatten().reshape(
            [-1, 3]).astype(np.uint8)
        image = tvtk.ImageData()
        image.point_data.scalars = image_data
        image.dimensions = self.texture.shape[1], self.texture.shape[0], 1
        texture = tvtk.Texture(input=image)
        actor.texture = texture
        self.figure.scene.add_actors(actor)

        return self


class MayaviVectorViewer3d(MayaviViewer):

    def __init__(self, figure_id, new_figure, points, vectors):
        super(MayaviVectorViewer3d, self).__init__(figure_id,
                                                   new_figure)
        self.points = points
        self.vectors = vectors

    def _render(self, **kwargs):
        # Only get every nth vector. 1 means get every vector.
        mask_points = kwargs.get('mask_points', 1)
        mlab.quiver3d(self.points[:, 0],
                      self.points[:, 1],
                      self.points[:, 2],
                      self.vectors[:, 0],
                      self.vectors[:, 1],
                      self.vectors[:, 2],
                      mask_points=mask_points,
                      figure=self.figure)
        return self