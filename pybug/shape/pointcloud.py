import numpy as np
from pybug.shape import Shape
from pybug.transform.base import Transformable
from pybug.shape.exceptions import PointFieldError
from pybug.visualize import PointCloudViewer3d, PointCloudViewer2d


class PointCloud(Shape, Transformable):
    """n-dimensional point cloud. Handles the addition of spatial
    metadata (most commonly landmarks) by storing all such 'metapoints'
    (points which aren't part of the shape) and normal points together into
    a joint field (points_and_metapoints). This is masked from the end user
    by the use of properties.
    """

    def __init__(self, points):
        super(PointCloud, self).__init__()
        self.n_points, n_dims = points.shape
        self.n_metapoints = 0
        cachesize = 1000
        self._allpoints = np.empty((self.n_points + cachesize, n_dims))
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
        don't simply reference existing points).
        """
        return self._allpoints[self.n_points:self.n_points_and_metapoints]

    @property
    def points_and_metapoints(self):
        return self._allpoints[:self.n_points_and_metapoints]

    @property
    def n_points_and_metapoints(self):
        return self.n_points + self.n_metapoints

    @property
    def n_dims(self):
        return self.points.shape[1]

    def as_vector(self):
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

    def view(self):
        if self.n_dims == 3:
            viewer = PointCloudViewer3d(self.points)
            return viewer.view()
        elif self.n_dims == 2:
            viewer = PointCloudViewer2d(self.points)
            return viewer
        else:
            print 'arbitrary dimensional PointCloud rendering is not ' \
                  'supported.'

    @property
    def _n_landmarkable_items(self):
        return self.n_points

    def _landmark_at_index(self, index):
        return self.points[index]

    def _add_meta_landmark_item(self, metapoint):
        """Adds a new metapoint to the cloud. Returns the index
        position that this point is stored at in self.metapoints.
        """
        if metapoint.size != self.n_dims:
            raise Exception("metapoint must be of the same number of dims "
                            "as the parent shape")
        next_index = self.n_points_and_metapoints
        self._allpoints[next_index] = metapoint.flatten()
        metapoint_index = self.n_metapoints
        self.n_metapoints += 1
        return metapoint_index

    def _meta_landmark_at_meta_index(self, meta_index):
        """
        Returns the metapoint at the meta meta_index.
        """
        return self.metapoints[meta_index]

    def _transform(self, transform):
        self._allpoints = transform(self._allpoints)
