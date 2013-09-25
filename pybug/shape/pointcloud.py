import numpy as np
from scipy.spatial.distance import cdist
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

    @property
    def centre(self):
        r"""
        The mean of all the points in this PointCloud.

        :type: (D,) ndarray
            The centre of this PointCloud.
        """
        return np.mean(self.points, axis=0)

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
        Builds a new :class:`PointCloud` given then ``flattened`` vector.
        This allows rebuilding pointclouds with the correct number of
        dimensions from a vector.

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

    def bounds(self, boundary=0):
        r"""
        The minimum to maximum extent of the :class:`PointCloud`.
        An optional boundary argument can be provided to expand the bounds
        by a constant margin.

        Parameters
        ----------
        boundary: b float
            A optional padding distance that is added to the bounds. Default
            is zero, meaning the max/min of tightest possible containing
            square/cube/hypercube is returned.

        Returns
        --------
        min_b : (D,) ndarray
            The minimum extent of the :class:`PointCloud` and boundary along
            each dimension

        max_b : (D,) ndarray
            The maximum extent of the :class:`PointCloud` and boundary along
            each dimension
        """
        min_b = np.min(self.points, axis=0) - boundary
        max_b = np.max(self.points, axis=0) + boundary
        return min_b, max_b

    def range(self, boundary=0):
        r"""
        The range of the extent of the :class:`PointCloud`.

        Parameters
        ----------
        boundary: b float
            A optional padding distance that is used to extend the bounds
            from which the range is computed. Default is zero, no extension
            is performed.

        Returns
        --------
        range : (D,) ndarray
            The range of the :class:`PointCloud`s extent in each dimension.
        """
        min_b, max_b = self.bounds(boundary)
        return max_b - min_b

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
        return PointCloudViewer(figure_id, new_figure,
                                self.points).render(**kwargs)

    def _transform_self(self, transform):
        self.points = transform(self.points)
        return self

    def distance_to(self, pointcloud, **kwargs):
        r"""
        Returns a distance matrix between this point cloud and another.
        By default the Euclidian distance is calculated - see
        ``scipy.spatial.distance.cdist`` for valid kwargs to change the metric
        and other properties.

        Parameters
        ----------
        pointcloud : PointCloud (M points, D dim)
            The second pointcloud to compute distances between. This must be
             of the same dimension as this PointCloud.

        Returns
        -------
        distance_matrix: (N, M) ndarray
            The symmetric pairwise distance matrix between the two PointClouds
            s.t. distance_matrix[i, j] is the distance between the i'th
            point of this PointCloud and the j'th point of the input
            PointCloud.
        """
        if self.n_dims != pointcloud.n_dims:
            raise ValueError("The two PointClouds must be of the same "
                             "dimensionality.")
        return cdist(self.points, pointcloud.points, **kwargs)

    def norm(self, **kwargs):
        r"""
        Returns the norm of this point cloud. This is a translation and
        rotation invariant measure of the point cloud's intrinsic size - in
        other words, it is always taken around the point cloud's centre.

        By default, the Frobenius norm is taken, but this can be changed by
        setting kwargs - see numpy.linalg.norm for valid options.

        Returns
        -------
        norm: float
            The norm of this :class:`PointCloud`
        """
        return np.linalg.norm(self.points - self.centre, **kwargs)
