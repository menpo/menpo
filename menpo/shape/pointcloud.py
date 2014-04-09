import numpy as np
from scipy.spatial.distance import cdist
from menpo.visualize import PointCloudViewer
from menpo.shape.base import Shape


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
        self.points = np.array(points, copy=True, order='C')

    @property
    def h_points(self):
        r"""
        homogeneous points of shape (n_dims + 1, n_points)
        """
        return np.concatenate((self.points.T, np.ones(self.n_points)[None, :]))

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
        The mean of all the points in this PointCloud (in the centre of mass
        sense)

        :type: (D,) ndarray
            The mean of this PointCloud's points.
        """
        return np.mean(self.points, axis=0)

    @property
    def centre_of_bounds(self):
        r"""
        The centre of the absolute bounds of this PointCloud. Contrast with
        centre, which is the mean point position.

        :type: (D,) ndarray
            The centre of the bounds of this PointCloud.
        """
        min_b, max_b = self.bounds()
        return (min_b + max_b) / 2

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

    def tojson(self):
        r"""
        Convert this `PointCloud` to a dictionary JSON representation.

        Returns
        -------
        dict with a 'points' key, the value of which is a list suitable
        for use in the by the `json` standard library package.
        """
        return {'points': self.points.tolist()}

    def from_vector_inplace(self, vector):
        r"""
        Updates this PointCloud in-place with a new vector of parameters
        """
        self.points = vector.reshape([-1, self.n_dims])

    def __str__(self):
        return '{}: n_points: {}, n_dims: {}'.format(type(self).__name__,
                                                     self.n_points,
                                                     self.n_dims)

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

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return PointCloudViewer(figure_id, new_figure,
                                self.points).render(**kwargs)

    def _transform_self_inplace(self, transform):
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
        pointcloud : :class:`PointCloud`
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

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the pointcloud. This is then broadcast across the dimensions
        of the pointcloud and returns a new pointcloud containing only those
        points that were `True` in the mask.

        Parameters
        ----------
        mask : (N,) ndarray
            1D array of booleans

        Returns
        -------
        pointcloud : :class:`PointCloud`
            A new pointcloud that has been masked.
        """
        return PointCloud(self.points[mask, :])

    def update_from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the pointcloud. This is then broadcast across the dimensions
        of the pointcloud. The same pointcloud is updated in place.

        Parameters
        ----------
        mask : (N,) ndarray
            1D array of booleans

        Returns
        -------
        pointcloud : :class:`PointCloud`
            A pointer to self.
        """
        self.points = self.points[mask, :]
        return self
