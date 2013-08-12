from pybug.exceptions import DimensionalityError
from pybug.visualize.base import Viewer


class Viewer2d(Viewer):
    """
    A viewer restricted to 2 dimensional data.
    """

    def __init__(self, points):
        Viewer.__init__(self)
        dim = points.shape[1]
        if dim != 2:
            raise DimensionalityError("Expected a 2-dimensional object, "
                                      "but got a {0} object. "
                                      "Provide an Nx2 object."
                                      .format(str(points.shape)))
        self.points = points

    @property
    def n_points(self):
        return self.points.shape[0]


class PointCloudViewer2d(Viewer2d):

    def __init__(self, points):
        super(PointCloudViewer2d, self).__init__(points)


class TriMeshViewer2d(PointCloudViewer2d):

    def __init__(self, points, trilist):
        super(TriMeshViewer2d, self).__init__(points)
        self.trilist = trilist


class LandmarkViewer2d(Viewer):

    def __init__(self, label, landmark_dict):
        super(LandmarkViewer2d, self).__init__()
        self.landmark_dict = landmark_dict
        self.label = label