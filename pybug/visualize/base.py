# This has to go above the default importers to prevent cyclical importing
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
        return self._viewonfigure(figure, **kwargs)


from pybug.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLabelViewer3d
from pybug.visualize.viewmatplotlib import MatplotLibImageViewer, \
    MatplotLibPointCloudViewer2d

# Default importer types
PointCloudViewer2d = MatplotLibPointCloudViewer2d
PointCloudViewer3d = MayaviPointCloudViewer3d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
LabelViewer3d = MayaviLabelViewer3d
ImageViewer = MatplotLibImageViewer
