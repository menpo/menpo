import numpy as np
from mayavi import mlab
from collections import OrderedDict

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
        self.n_metapoints = n_metapoints
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
    Landmarks index into the points and metapoints of the associated 
    PointCloud. Landmarks which are expicitly given as coordinates would
    be entirely constructed from metapoints, whereas point indexed landmarks
    would be composed entirely of points. This class can handle any arbitrary
    mixture of the two.
    """
    def __init__(self, pointcloud, landmark_dict):
        """ pointcloud - the shape whose these landmarks apply to
        landmark_dict - keys - landmark classes (e.g. 'mouth')
                        values - ordered list of landmark indices into pointcloud._allpoints
        """
        self.pc = pointcloud
        # indexes are the indexes into the points and metapoints of self.pc.
        # note that the labels are always sorted when stored.
        self.indexes = OrderedDict(sorted(landmark_dict.iteritems()))

    def all(self, withlabels=False):
        """return all the landmark indexes. The order is always guaranteed to
        be the same for a given landmark configuration - specifically, the points
        will be returned by sorted label, and always in the order that each point 
        the landmark was construted in.
        """
        all_lm = []
        labels = []
        for k, v in self.indexes.iteritems():
            all_lm += v
            labels += [k] * len(v)
        if withlabels:
           return all_lm, labels
        return all_lm

    @property
    def config(self):
        """A nested tuple specifying the precise nature of the landmarks
        (labels, and n_points per label). Allows for comparison of Landmarks
        to see if they are likely describing the same shape.
        """
        return tuple((k,len(v)) for k,v in self.indexes.iteritems())
       
