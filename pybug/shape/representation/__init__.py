import numpy as np
from mayavi import mlab

class PointFieldError(Exception):
    pass




class Shape(object):
    """ Abstract representation of a n-dimentional shape. This could be simply
    be a set of vectors in an n-dimentional shape. Optionally, all shapes can
    have associated with them a Landmarks object containing annotations about
    the object in the space.
    """
    def __init__(self):
        pass


class PointCloud(Shape):
    """n-dimensional point cloud. Can be coerced to a PCL Point Cloud Object
    for using their library methods.
    """
    def __init__(self, points, n_metapoints=0):
        Shape.__init__(self)
        self.n_points, n_dims  = points.shape
        self._allpoints = np.empty([self.n_points + self.n_metapoints, n_dims])
        self._allpoints[:self.n_points] = points
        self.pointfields = {}

    @property
    def points(self):
        return self._allpoints[:self.n_points]

    @property
    def metapoints(self):
        """Points which are solely for metadata. Are guaranteed to be
        transformed in exactly the same way that points are. Useful for
        storing explicit landmarks (landmarks that have coordinates and
        don't simply reference exisiting points).
        """
        return self._allpoints[self.n_points:]

    @property
    def n_dims(self):
        return self.points.shape[1]

    def __str__(self):
        message = str(type(self)) + ': n_points: ' + `self.n_points`  \
                + ', n_dims: ' + `self.n_dims`
        if len(self.pointfields) != 0:
            message += '\n  pointfields:'
            for k,v in self.pointfields.iteritems():
                try:
                    field_dim = v.shape[1]
                except IndexError:
                    field_dim = 1
                message += '\n    ' + str(k) + '(' + str(field_dim) + 'D)'
        return message


    def add_pointfield(self, name, field):
        if field.shape[0] != self.n_points:
            raise PointFieldError("Trying to add a field with " +
                    `field.shape[0]` + " values (need one field value per " +
                    "point => " + `self.n_points` + " values required")
        else:
            self.pointfields[name] = field

    def view(self):
        if self.n_dims == 3:
            f =mlab.figure()
            f.scene.background = (1,1,1)
            mlab.points3d(self.points[:,0], self.points[:,1], self.points[:,2],
                    mode='sphere', color=(0,0,0), figure=f)
            return f
        else:
            print 'only 3D PointCloud rendering is supported at this time.'

class Landmarks(object):
    """Class for storing and manipulating Landmarks associated with a shape.
    Landmarks are named sets of annotations. They inherit from PointCloud so
    as to have access to the full set of PointCloud operations, but impliment
    a different constructor.
    """
    def __init__(self, points):
        Shape.__init__(self)
        self.points = np.array(points, dtype=np.float)

