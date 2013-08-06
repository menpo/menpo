import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pybug.visualize import view2d


class MatplotlibViewer(object):
    def newfigure(self):
        return plt.figure()


def MatplotLibImageViewer(image):
    if image.shape[2] == 1:
        return plt.imshow(image[..., 0], cmap=cm.Greys_r)
    else:
        return plt.imshow(image)


class MatplotLibPointCloudViewer2d(view2d.PointCloudViewer2d, MatplotlibViewer):

    def __init__(self, points):
        view2d.PointCloudViewer2d.__init__(self, points)

    def _viewonfigure(self, figure, **kwargs):
        style = kwargs.pop('style', 'bo')
        self.currentscene = figure.add_subplot(111).plot(self.points[:, 0],
                                                         self.points[:, 1],
                                                         style, **kwargs)
        self.currentfigure = figure
        return self