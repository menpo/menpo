import numpy as np

class Viewer3dError(Exception):
    pass

class Viewer(object):
    """Abstract class for performaing visualizations. Framework specific 
    implimentations of these classes are made in order to seperate implimentation
    cleanly from the rest of the code.
    """

    def __init__(self):
        self.currentfig = None

class Viewer3d(Viewer):

    def __init__(self, points):
        Viewer.__init__(self)
        dim = points.shape[1]
        if dim != 3:
            raise Viewer3dError("Trying to view " + str(dim) +\
                    "data with a 3DViewer")
        self.points = points


class PointCloudViewer3d(Viewer3d):

    def __init__(self, points):
        Viewer3d.__init__(self, points)



