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
            ax.imshow(self.image[..., 0], cmap=cm.Greys_r, **kwargs)
        else:
            ax.imshow(self.image, **kwargs)

        return self


class MatplotlibPointCloudViewer2d(view2d.PointCloudViewer2d,
                                   MatplotlibViewer):

    def __init__(self, points):
        view2d.PointCloudViewer2d.__init__(self, points)

    def _viewonfigure(self, figure, **kwargs):
        self.currentfigure = figure

        figure.add_subplot(111).scatter(self.points[:, 0],
                                        self.points[:, 1],
                                        **kwargs)
        return self


class MatplotlibLandmarkViewer2d(view2d.LandmarkViewer2d, MatplotlibViewer):

    def __init__(self, label, landmark_dict):
        view2d.LandmarkViewer2d.__init__(self, label, landmark_dict)

    def _viewonfigure(self, figure, **kwargs):
        self.currentfigure = figure

        cmap = kwargs.get('colormap', cm.jet)
        halign = kwargs.get('halign', 'center')
        valign = kwargs.get('valign', 'bottom')
        colours = kwargs.get('colours',
                             np.random.random([3, len(self.landmark_dict)]))

        for i, (label, pcloud) in enumerate(self.landmark_dict.iteritems()):
            colour_array = [colours[:, i]] * pcloud.n_points
            pcloud.view(onviewer=self, cmap=cmap, c=colour_array, **kwargs)

            points = pcloud.points
            for i, p in enumerate(points):
                msg = '{0}_{1}_{2}'.format(self.label, label, i)
                figure.add_subplot(111).annotate(msg, xy=(p[0], p[1]),
                                                 horizontalalignment=halign,
                                                 verticalalignment=valign)
        return self