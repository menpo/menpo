import numpy as np
from warnings import warn
from scipy.spatial.distance import cdist
from menpo.visualize import PointCloudViewer
from menpo.shape.base import Shape
PointDirectedGraph = None

_bounding_box_adj = np.array([[0, 3], [2, 0], [1, 2], [1, 3]])


class PointCloud(Shape):
    r"""
    An N-dimensional point cloud. This is internally represented as an ndarray
    of shape (``n_points``, ``n_dims``). This class is important for dealing
    with complex functionality such as viewing and representing metadata such
    as landmarks.

    Currently only 2D and 3D pointclouds are viewable.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The array representing the points.

    copy : `boolean`, optional
        If ``False``, the points will not be copied on assignment. Note that
        this will miss out on additional checks. Further note that we still
        demand that the array is C-contiguous - if it isn't, a copy will be
        generated anyway.
        In general this should only be used if you know what you are doing.
    """

    def __init__(self, points, copy=True):
        super(PointCloud, self).__init__()
        if not copy:
            if not points.flags.c_contiguous:
                warn('The copy flag was NOT honoured. A copy HAS been made. '
                     'Please ensure the data you pass is C-contiguous.')
                points = np.array(points, copy=True, order='C')
        else:
            points = np.array(points, copy=True, order='C')
        self.points = points

    @property
    def n_points(self):
        r"""
        The number of points in the pointcloud.

        :type: `int`
        """
        return self.points.shape[0]

    @property
    def n_dims(self):
        r"""
        The number of dimensions in the pointcloud.

        :type: `int`
        """
        return self.points.shape[1]

    def h_points(self):
        r"""
        homogeneous points of shape (``n_dims + 1``, ``n_points``)
        """
        return np.concatenate((self.points.T, np.ones(self.n_points)[None, :]))

    def centre(self):
        r"""
        The mean of all the points in this PointCloud (in the centre of mass
        sense)

        Returns
        -------
        centre : ``(n_dims)`` `ndarray`
            The mean of this PointCloud's points.
        """
        return np.mean(self.points, axis=0)

    def centre_of_bounds(self):
        r"""
        The centre of the absolute bounds of this PointCloud. Contrast with
        centre, which is the mean point position.

        Returns
        -------
        centre : ``n_dims`` `ndarray`
            The centre of the bounds of this PointCloud.
        """
        return self.bounds().centre()

    def _as_vector(self):
        r"""
        Returns a flattened representation of the pointcloud.
        Note that the flattened representation is of the form
        ``[x0, y0, x1, y1, ....., xn, yn]`` for 2D.

        Returns
        -------
        flattened : ``(n_points,)`` `ndarray`
            The flattened points.
        """
        return self.points.ravel()

    def tojson(self):
        r"""
        Convert this PointCloud to a dictionary JSON representation.

        Returns
        -------
        json_dict : `dict`
            Dictionary with a 'points' key, the value of which is a list
            suitable for use in the by the `json` standard library package.
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
        The minimum to maximum extent of the :map:`PointCloud`.
        An optional boundary argument can be provided to expand the bounds
        by a constant margin.

        Parameters
        ----------
        boundary : `float`
            A optional padding distance that is added to the bounds. Default
            is ``0``, meaning the max/min of tightest possible containing
            square/cube/hypercube is returned.

        Returns
        --------
        min_b : ``(n_dims,)`` `ndarray`
            The minimum extent of the :map:`PointCloud` and boundary along
            each dimension

        max_b : ``(n_dims,)`` `ndarray`
            The maximum extent of the :map:`PointCloud` and boundary along
            each dimension
        """
        min_b = np.min(self.points, axis=0) - boundary
        max_b = np.max(self.points, axis=0) + boundary
        return BoundingBox(np.array((min_b, max_b)))

    def range(self, boundary=0):
        r"""
        The range of the extent of the :map:`PointCloud`.

        Parameters
        ----------
        boundary : `float`
            A optional padding distance that is used to extend the bounds
            from which the range is computed. Default is ``0``, no extension
            is performed.

        Returns
        -------
        range : ``(n_dims,)`` `ndarray`
            The range of the :map:`PointCloud` extent in each dimension.
        """
        bb = self.bounds(boundary)
        return bb.max - bb.min

    def view(self, figure_id=None, new_figure=False, **kwargs):
        return PointCloudViewer(figure_id, new_figure,
                                self.points).render(**kwargs)

    def _transform_self_inplace(self, transform):
        self.points = transform(self.points)
        return self

    def distance_to(self, pointcloud, **kwargs):
        r"""
        Returns a distance matrix between this point cloud and another.
        By default the Euclidian distance is calculated - see
        `scipy.spatial.distance.cdist` for valid kwargs to change the metric
        and other properties.

        Parameters
        ----------
        pointcloud : :map:`PointCloud`
            The second pointcloud to compute distances between. This must be
            of the same dimension as this PointCloud.

        Returns
        -------
        distance_matrix: ``(n_points, n_points)`` `ndarray`
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
        norm : `float`
            The norm of this :map:`PointCloud`
        """
        return np.linalg.norm(self.points - self.centre(), **kwargs)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the pointcloud. This is then broadcast across the dimensions
        of the pointcloud and returns a new pointcloud containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        pointcloud : :map:`PointCloud`
            A new pointcloud that has been masked.

        Raises
        ------
        ValueError
            Mask must have same number of points as pointcloud.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this PointCloud.')
        pc = self.copy()
        pc.points = pc.points[mask, :]
        return pc


class BoundingBox(PointCloud):
    r"""
    An N-dimensional bounding box built from 2 points - the minimum point of
    the box, and the maximum of the box. Only the two points are stored, but
    the a full :map:`PointDirectedGraph` representation of the box
    (four points in the case of 2D) can be retrieved using the `.box()` method.


    Parameters
    ----------
    min_max : ``(2, n_dims)`` `ndarray`
        The array representing the minimum and maximum points defining the
        bounding box

    skip_checks : `boolean`, optional
        If ``False``, we will check if only 2 points were provided. If ``True``
        this skip will be checked - only useful if you are generating
        :map:`BoundingBox` instances in a tight loop.

    """
    def __init__(self, min_max, skip_checks=False):
        PointCloud.__init__(self, min_max)
        if not skip_checks:
            if self.n_points != 2:
                raise ValueError(
                    'Bounding box should be built from just the '
                    'min and max points (so a numpy array of shape '
                    '(2, n_dims)')

    def __str__(self):
        return ('{}: min: {}, '
                'max: {}, n_dims: {}'.format(type(self).__name__, self.min,
                                             self.max, self.n_dims))

    @property
    def min(self):
        r"""
        The minimum of the bounding box extent.

        :type: `ndarray shape (n_dims,)`
        """
        return self.points[0]

    @property
    def max(self):
        r"""
        The maximum of the bounding box extent.

        :type: `ndarray shape (n_dims,)`
        """
        return self.points[1]

    def box(self):
        r"""
        Generate a full box :map:`PointDirectedGraph` representation of this
        :map:`BoundingBox`. Will have 4 points in the case of 2D. Other
        dimensions are not yet supported.

        Returns
        -------
        pointgraph : :map:`PointDirectedGraph`
            A 'complete' representation of this :map:`BoundingBox`

        """
        global PointDirectedGraph
        if PointDirectedGraph is None:
            from .graph import PointDirectedGraph
        if self.n_dims != 2:
            raise ValueError('BoundingBox.box() does not support dimensions '
                             'other than 2 (yet)')
        p = self.points
        p2 = [[p[0, 0], p[1, 1]],
              [p[1, 0], p[0, 1]]]
        points = np.vstack((p, p2))
        return PointDirectedGraph(points, _bounding_box_adj.copy(), copy=False)

    def _view(self, figure_id=None, new_figure=False, **kwargs):
        return self.box()._view(figure_id=None, new_figure=False, **kwargs)
