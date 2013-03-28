import numpy as np

class PointFieldError(Exception):
    pass


class Landmarks(object):
    """Class for storing and manipulating Landmarks associated with a shape.
    Landmarks are named sets of annotations.
    """
    def __init__(self):
        pass


class Shape(object):
    """ Abstract representation of a n-dimentional shape. This could be simply
    be a set of vectors in an n-dimentional shape. Optionally, all shapes can
    have associated with them a Landmarks object containing annotations about
    the object in the space.
    """
    def __init__(self):
        self.landmarks = Landmarks()


class PointCloud(Shape):
    """n-dimensional point cloud. Can be coerced to a PCL Point Cloud Object
    for using their library methods.
    """
    def __init__(self, points):
        Shape.__init__(self)
        self.points = np.array(points, dtype=np.float)
        self.pointfields = {}

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def n_dims(self):
        return self.points.shape[1]

    def __str__(self):
        message = 'PointCloud: n_points: ' + `self.n_points`  \
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

