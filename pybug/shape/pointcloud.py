from pybug.shape import Shape
from pybug.shape.exceptions import PointFieldError
from pybug.visualize import PointCloudViewer


# TODO: sort of pointfields?
class PointCloud(Shape):
    r"""
    An N-dimensional point cloud. This is internally represented as an ndarray
    of shape (``n_points``, ``n_dims``). This class is important for dealing
    with complex functionality such as viewing and representing metadata such
    as landmarks.

    Currently only 2D and 3D pointclouds are viewable.

    Parameters
    ----------
    points : (N, D) ndarray
        A (``n_points``, ``n_dims``) ndarray representing the points.
    """

    def __init__(self, points):
        super(PointCloud, self).__init__()
        self.points = points
        self.pointfields = {}

    @property
    def n_points(self):
        r"""
        The number of points in the pointcloud.

        :type: int
        """
        return self.points.shape[0]

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the pointcloud.

        :type: int
        """
        return self.points.shape[1]

    def as_vector(self):
        r"""
        Returns a flattened representation of the pointcloud.
        Note that the flattened representation is of the form
        ``[x0, y0, x1, y1, ....., xn, yn]`` for 2D.

        Returns
        -------
        flattened : (N,) ndarray
            The flattened points.
        """
        return self.points.flatten()

    def from_vector(self, flattened):
        r"""
        Builds a new pointcoloud given then ``flattened`` vector. This allows
        rebuilding pointclouds with the correct number of dimensions from a
        vector.

        Parameters
        ----------
        flattened : (N,) ndarray
            Vector representing a set of points.

        Returns
        --------
        pointcloud : :class:`PointCloud`
            A new pointcloud created from the vector.
        """
        return PointCloud(flattened.reshape([-1, self.n_dims]))

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
        """
        Add another set of field values (of arbitrary dimension) to each
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

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return PointCloudViewer(figure_id, new_figure, self.points).render(**kwargs)

    def _transform_self(self, transform):
        self.points = transform(self.points)
        return self
