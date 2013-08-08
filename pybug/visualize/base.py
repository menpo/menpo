# This has to go above the default importers to prevent cyclical importing
from pybug.exceptions import DimensionalityError


class Viewer(object):
    """Abstract class for performing visualizations. Framework specific
    implementations of these classes are made in order to separate
    implementation cleanly from the rest of the code.
    """

    def __init__(self):
        self.currentfigure = None
        self.currentscene = None

    def view(self, **kwargs):
        figure = kwargs.pop('onviewer', None)
        if figure is None:
            figure = self.newfigure()
        else:
            figure = figure.currentfigure
        return self._viewonfigure(figure, **kwargs)

from pybug.visualize.viewmayavi import MayaviPointCloudViewer3d, \
    MayaviTriMeshViewer3d, MayaviTexturedTriMeshViewer3d, \
    MayaviLandmarkViewer3d
from pybug.visualize.viewmatplotlib import MatplotlibImageViewer2d, \
    MatplotlibPointCloudViewer2d, MatplotlibLandmarkViewer2d, \
    MatplotlibLandmarkViewer2dImage

# Default importer types
PointCloudViewer2d = MatplotlibPointCloudViewer2d
PointCloudViewer3d = MayaviPointCloudViewer3d
TriMeshViewer3d = MayaviTriMeshViewer3d
TexturedTriMeshViewer3d = MayaviTexturedTriMeshViewer3d
LandmarkViewer3d = MayaviLandmarkViewer3d
LandmarkViewer2d = MatplotlibLandmarkViewer2d
LandmarkViewer2dImage = MatplotlibLandmarkViewer2dImage
ImageViewer2d = MatplotlibImageViewer2d


class LandmarkViewer(Viewer):
    """
    Base Landmark viewer that abstracts away dimensionality
    """
    def __init__(self, label, landmark_dict, parent_shape):
        Viewer.__init__(self)
        if landmark_dict is None:
            landmark_dict = {}
        self.landmark_dict = landmark_dict
        self.label = label
        self.shape = parent_shape

    def _viewonfigure(self, figure, **kwargs):
        self.currentfigure = figure

        if self.landmark_dict:
            item = self.landmark_dict.values()[0]
            if item.n_dims == 2:
                from pybug.image import Image
                if type(self.shape) is Image:
                    return LandmarkViewer2dImage(
                        self.label, self.landmark_dict).view(onviewer=self,
                                                             **kwargs)
                else:
                    return LandmarkViewer2d(
                        self.label, self.landmark_dict).view(onviewer=self,
                                                             **kwargs)
            elif item.n_dims == 3:
                return LandmarkViewer3d(self.label, self.landmark_dict).view(
                    onviewer=self, **kwargs)
            else:
                raise DimensionalityError("Only 2D and 3D landmarks are "
                                          "currently supported")


