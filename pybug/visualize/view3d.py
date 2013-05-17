import numpy as np


class Viewer3dError(Exception):
    pass


class TCoordsViewerError(Viewer3dError):
    pass


class Viewer(object):
    """Abstract class for performing visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.
    """

    def __init__(self):
        self.currentfigure = None
        self.currentscene = None

    def view(self, **kwargs):
        figure = kwargs.get('onviewer')
        if figure is None:
            figure = self.newfigure()
        else:
            figure = figure.currentfigure
            print figure
        return self._viewonfigure(figure, **kwargs)


class Viewer3d(Viewer):
    """ A viewer restricted to 3 dimensional data.
    """

    def __init__(self, points):
        Viewer.__init__(self)
        dim = points.shape[1]
        if dim != 3:
            raise Viewer3dError("Expected a 3-dimensional object, "
                                "but got a {0} object. "
                                "Provide an Nx3 object."
                                .format(str(points.shape)))
        self.points = points

    @property
    def n_points(self):
        return self.points.shape[0]


class PointCloudViewer3d(Viewer3d):
    def __init__(self, points):
        Viewer3d.__init__(self, points)


class LabelViewer3d(Viewer3d):
    def __init__(self, points, labels, **kwargs):
        Viewer3d.__init__(self, points)
        self.labels = labels
        if len(self.labels) != self.n_points:
            raise Viewer3dError('Must have len(labels) == n_points')
        offset = kwargs.get('offset', np.zeros(3))
        if offset.shape == (3,):
            #single offset provided -> apply to all
            self.offset = np.repeat(offset[np.newaxis, :],
                                    self.n_points, axis=0)
        elif offset.shape == (3, self.n_points):
            self.offset = offset
        else:
            raise Viewer3dError('Offset must be of shape (3,)'
                                + 'or (3,n_points) if specified')


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
        if self.tcoords_per_tri is None and self.tcoords_per_point is None:
            raise TCoordsViewerError("tcoords need to be provided per-point" +
                                     "or per-triangle")
        if self.tcoords_per_tri is not None:
            # for now we don't render these well, and just convert to a tcoord
            # per point representation.
            self.tcoords_per_point = self.tc_per_tri_to_tc_per_point(
                self.trilist, self.tcoords_per_tri)

    def tc_per_tri_to_tc_per_point(self, trilist, tcoords):
        """Generates per-point tc from per-tri tc. Obviously
         this means we loose data (some triangles will have fugly
         textures) but allows for quick and dirty visualize of textures.
        """
        u_ci, ind_of_u_ci = np.unique(trilist, return_index=True)
        return tcoords.reshape([-1, 2])[ind_of_u_ci]

