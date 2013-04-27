import numpy as np
from pybug.spatialdata import SpatialData
from pybug.spatialdata.landmarks import LandmarkManager
from pybug.spatialdata.exceptions import PointFieldError
from pybug.visualization import PointCloudViewer3d


class PointCloud(SpatialData):
    """n-dimensional point cloud. Handles the addition of spatial
    metadata (most commonly landmarks) by storing all such 'metapoints'
    (points which aren't part of the shape) and normal points together into
    a joint field (points_and_metapoints). This is masked from the end user
    by the use of properties.
    """

    def __init__(self, points):
        SpatialData.__init__(self)
        self.n_points, n_dims = points.shape
        self.n_metapoints = 0
        cachesize = 1000
        # preallocate allpoints to have enough room for cachesize metapoints
        self._allpoints = np.empty((self.n_points + cachesize, n_dims))
        self._allpoints[:self.n_points] = points
        self.pointfields = {}
        self.landmarks = LandmarkManager(self)

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
        return self._allpoints[self.n_points:]

    @property
    def points_and_metapoints(self):
        return self._allpoints[:self.n_points_and_metapoints]

    @property
    def n_points_and_metapoints(self):
        return self.n_points + self.n_metapoints

    @property
    def n_dims(self):
        return self.points.shape[1]

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

    def add_metapoint(self, metapoint):
        """Adds a new metapoint to the pointcloud. Returns the index
        position that this point is stored at in self.points_and_metapoints.
        """
        if metapoint.size != self.n_dims:
            raise Exception("metapoint must be of the same number of dims "
                            "as the parent pointcloud")
        next_index = self.n_points_and_metapoints
        self._allpoints[next_index] = metapoint.flatten()
        self.n_metapoints += 1
        return next_index

    def view(self):
        if self.n_dims == 3:
            viewer = PointCloudViewer3d(self.points, **kwargs)
            return viewer.view()
        else:
            print 'arbitrary dimensional PointCloud rendering is not supported.'
