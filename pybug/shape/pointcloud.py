import numpy as np
from pybug.shape import Shape
from pybug.shape.exceptions import PointFieldError
from pybug.visualize import PointCloudViewer3d, PointCloudViewer2d


class PointCloud(Shape):
    """
    N-dimensional point cloud.
    """

    def __init__(self, points):
        super(PointCloud, self).__init__()
        self.points = points
        self.pointfields = {}

    @property
    def n_points(self):
        return self.points.shape[0]

    @property
    def n_dims(self):
        return self.points.shape[1]

    def as_vector(self):
        """
        Returns a flattened representation of the pointcloud.
        Note that the flattened representation is of the form
        [x0, y0, x1, y1, ....., xn, yn] for 2D.
        :return:
        """
        return self.points.flatten()

    def from_vector(self, flattened):
        n_dims = self.n_dims
        return PointCloud(flattened.reshape([-1, n_dims]))

    def __str__(self):
        message = (str(type(self)) + ': n_points: ' + str(self.n_points) +
                   ', n_dims: ' + str(self.n_dims))
        if len(self.pointfields) != 0:
            message += '\n  pointfields:'
            for k, v in self.pointfields.iteritems():
                try:
                    field_dim = v.shape[1]
                except IndexError:
                    field_dim = 1
                message += '\n    ' + str(k) + '(' + str(field_dim) + 'D)'
        return message

    def add_pointfield(self, name, field):
        """Add another set of field values (of arbitrary dimension) to each
        point.
        """
        if field.shape[0] != self.n_points:
            raise PointFieldError("Trying to add a field with " +
                                  str(field.shape[0]) + " values (need one "
                                                        "field value per point"
                                                        " => "
                                  + str(self.n_points) + " values required")
        else:
            self.pointfields[name] = field

    def view(self, **kwargs):
        if self.n_dims == 3:
            return PointCloudViewer3d(self.points, **kwargs).view(**kwargs)
        elif self.n_dims == 2:
            return PointCloudViewer2d(self.points).view(**kwargs)
        else:
            print 'arbitrary dimensional PointCloud rendering is not ' \
                  'supported.'

    def _transform_self(self, transform):
        self.points = transform(self.points)
        return self
