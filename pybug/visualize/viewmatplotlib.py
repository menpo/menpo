import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pybug.visualize import view2d, viewimage


class MatplotlibViewer(object):
    def newfigure(self):
        return plt.figure()


class MatplotlibImageViewer2d(viewimage.ImageViewer2d, MatplotlibViewer):

    def __init__(self, image):
        viewimage.ImageViewer2d.__init__(self, image)

    def _viewonfigure(self, figure, **kwargs):
        self.currentfigure = figure
        ax = figure.add_subplot(111)

        if self.image.shape[2] == 1:
            self.currentscene = ax.imshow(self.image[..., 0],
                                          cmap=cm.Greys_r, **kwargs)
        else:
            self.currentscene = ax.imshow(self.image, **kwargs)

        return self


class MatplotlibPointCloudViewer2d(view2d.PointCloudViewer2d,
                                   MatplotlibViewer):

    def __init__(self, points):
        view2d.PointCloudViewer2d.__init__(self, points)

    def _viewonfigure(self, figure, image_view=False, cmap=None,
                      colour_array='b', label=None, **kwargs):
        self.currentfigure = figure

        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        self.currentscene = figure.add_subplot(111)
        self.currentscene.scatter(points[:, 0], points[:, 1], cmap=cmap,
                                  c=colour_array, label=label)
        return self


class MatplotlibTriMeshViewer2d(view2d.TriMeshViewer2d, MatplotlibViewer):

    def __init__(self, points, trilist):
        super(MatplotlibTriMeshViewer2d, self).__init__(points, trilist)

    def _viewonfigure(self, figure, image_view=False, **kwargs):
        self.currentfigure = figure

        # Flip x and y for viewing if points are tied to an image
        points = self.points[:, ::-1] if image_view else self.points
        self.currentscene = figure.add_subplot(111)

        for tri in self.trilist:
            joined_tri = np.append(tri, tri[0])
            self.currentscene.plot(points[joined_tri, 0],
                                   points[joined_tri, 1], **kwargs)

        return self


class MatplotlibLandmarkViewer2d(view2d.LandmarkViewer2d, MatplotlibViewer):

    def __init__(self, label, landmark_dict):
        view2d.LandmarkViewer2d.__init__(self, label, landmark_dict)

    def _plot_landmarks(self, include_labels, image_view, **kwargs):
        colours = kwargs.get('colours',
                             np.random.random([3, len(self.landmark_dict)]))
        halign = kwargs.get('halign', 'center')
        valign = kwargs.get('valign', 'bottom')
        size = kwargs.get('size', 10)

        # TODO: Should we enforce viewing landmarks with Matplotlib? How
        # do we do this?
        # Set the default colormap, assuming that the pointclouds are
        # also viewed using Matplotlib
        kwargs.setdefault('cmap', cm.jet)

        for i, (label, pc) in enumerate(self.landmark_dict.iteritems()):
            # Set kwargs assuming that the pointclouds are viewed using
            # Matplotlib
            kwargs['colour_array'] = [colours[:, i]] * pc.n_points
            kwargs['label'] = '{0}_{1}'.format(self.label, label)
            pc.view(image_view=image_view, **kwargs)

            if include_labels:
                ax = self.currentfigure.gca()
                points = pc.points[:, ::-1] if image_view else pc.points
                for i, p in enumerate(points):
                    ax.annotate(str(i), xy=(p[0], p[1]),
                                horizontalalignment=halign,
                                verticalalignment=valign, size=size)
                ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    def _viewonfigure(self, figure, include_labels=True, **kwargs):
        self.currentfigure = figure
        self._plot_landmarks(include_labels, False, **kwargs)
        return self


class MatplotlibLandmarkViewer2dImage(MatplotlibLandmarkViewer2d):

    def __init__(self, label, landmark_dict):
        MatplotlibLandmarkViewer2d.__init__(self, label, landmark_dict)

    def _viewonfigure(self, figure, include_labels=True, **kwargs):
        self.currentfigure = figure
        self._plot_landmarks(include_labels, True, **kwargs)
        return self