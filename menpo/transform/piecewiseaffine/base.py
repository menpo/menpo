import abc
import numpy as np
from copy import deepcopy
from menpo.base import Copyable
from menpo.transform.base import Alignment, Invertible, Transform
from .fastpwa import CLookupPWA
# TODO View is broken for PWA (TriangleContainmentError)


class TriangleContainmentError(Exception):
    r"""
    Exception that is thrown when an attempt is made to map a point with a
    PWATransform that does not lie in a source triangle.

    points_outside_source_domain : (d,) ndarray
        A boolean value for the d points that were attempted to be applied.
        If True, the point was outside of the domain.
    """
    def __init__(self, points_outside_source_domain):
        super(TriangleContainmentError, self).__init__()
        self.points_outside_source_domain = points_outside_source_domain


def containment_from_alpha_beta(alpha, beta):
    r"""
    Check `alpha` and `beta` are within a triangle (`alpha >= 0`,
    `beta >= 0`, `alpha + beta <= 1`). Returns the indices of the
    triangles that are `alpha` and `beta` are in. If any of the
    points are not contained in a triangle,
    raises a TriangleContainmentError.

    Parameters
    ----------
    alpha: (K, `n_tris`) ndarray
        Alpha for each point and triangle being tested.
    beta: (K, `n_tris`) ndarray
        Beta for each point and triangle being tested.

    Returns
    -------
    tri_index : (L,) ndarray
        triangle index for each `points`, assigning each
        point in a triangle to the triangle index.

    Raises
    ------
    TriangleContainmentError
    All `points` must be contained in a source triangle. Check
    `error.points_outside_source_domain` to handle this case.
    """
    # (K, n_tris), boolean for whether a given triangle contains a given
    #  point
    point_containment = np.logical_and(np.logical_and(
        alpha >= 0, beta >= 0),
        alpha + beta <= 1)
    # is each point in a triangle?
    point_in_a_triangle = np.any(point_containment, axis=1)
    if np.any(~point_in_a_triangle):
        raise TriangleContainmentError(~point_in_a_triangle)
    point_index, tri_index = np.nonzero(point_containment)
    # don't want duplicates! ensure that here:
    index = np.zeros(alpha.shape[0])
    index[point_index] = tri_index
    return index.astype(np.uint32)


def alpha_beta(i, ij, ik, points):
    r"""
    Calculates the alpha and beta values (barycentric coordinates) for each
    triangle for all points provided. Note that this does not raise a
    TriangleContainmentError.

    Parameters
    ----------
    i : (`n_tris`, 2) ndarray
        The coordinate of the i'th point of each triangle

    ij (`n_tris`, 2) ndarray
        The vector between the i'th point and the j'th point of each
        triangle

    ik (`n_tris`, 2) ndarray
        The vector between the i'th point and the k'th point of each
        triangle

    points : (`n_points`, 2) ndarray
        Points to calculate the barycentric coordinates for.

    Returns
    --------
    alpha : (`n_points`, `n_tris`)
        The alpha for each point and triangle. Alpha can be interpreted
        as the contribution of the ij vector to the position of the
        point in question.
    beta : (`n_points`, `n_tris`)
        The beta for each point and triangle. Beta can be interpreted as
         the contribution of the ik vector to the position of the point
         in question.
    """
    ip = points[..., None] - i
    dot_jj = np.einsum('dt, dt -> t', ij, ij)
    dot_kk = np.einsum('dt, dt -> t', ik, ik)
    dot_jk = np.einsum('dt, dt -> t', ij, ik)
    dot_pj = np.einsum('vdt, dt -> vt', ip, ij)
    dot_pk = np.einsum('vdt, dt -> vt', ip, ik)

    d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
    alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d
    beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d
    return alpha, beta


def index_alpha_beta(i, ij, ik, points):
    """
    Finds for each input point the index of it's bounding triangle
    and the alpha and beta value for that point in the triangle. Note
    this means that the following statements will always be true:
        alpha + beta <= 1
        alpha >= 0
        beta >= 0
    for each triangle result.
    Trying to map a point that does not exist in a
    triangle throws a TriangleContainmentError.

    Parameters
    ----------
    i : (`n_tris`, 2) ndarray
        The coordinate of the i'th point of each triangle

    ij (`n_tris`, 2) ndarray
        The vector between the i'th point and the j'th point of each
        triangle

    ik (`n_tris`, 2) ndarray
        The vector between the i'th point and the k'th point of each
        triangle

    points : (`n_points`, 2) ndarray
        Points to calculate the barycentric coordinates for.


    Returns
    -------
    tri_index : (`n_tris`,) ndarray
        triangle index for each of the `points`, assigning each
        point to it's containing triangle.
    alpha : (`n_tris`,) ndarray
        Alpha for containing triangle of each point.
    beta : (`n_tris`,) ndarray
        Beta for containing triangle of each point.

    Raises
    ------
    TriangleContainmentError
    All `points` must be contained in a source triangle. Check
    `error.points_outside_source_domain` to handle this case.
    """
    alpha, beta = alpha_beta(i, ij, ik, points)
    each_point = np.arange(points.shape[0])
    index = containment_from_alpha_beta(alpha, beta)
    return index, alpha[each_point, index], beta[each_point, index]


def barycentric_vectors(points, trilist):
    r"""
    Compute the affine transformation between each triangle in the source
    and target. This is calculated analytically.

    Parameters
    ----------

    points : (`n_points`, 2) ndarray
        Points to calculate the barycentric coordinates for.

    trilist: (`n_tris`, 3) ndarray
        The 0-based index triangulation joining the points.

    Returns
    -------
    i : (`n_tris`, 2) ndarray
        The coordinate of the i'th point of each triangle

    ij (`n_tris`, 2) ndarray
        The vector between the i'th point and the j'th point of each
        triangle

    ik (`n_tris`, 2) ndarray
        The vector between the i'th point and the k'th point of each
        triangle
    """
    # we permute the axes of the indexed point set to have shape
    # [3, n_dims, n_tris] for ease of indexing in.
    x = np.transpose(points[trilist],  axes=[1, 2, 0])
    return x[0], x[1] - x[0], x[2] - x[0]


# Note we inherit from Alignment first to get it's n_dims behavior
class AbstractPWA(Alignment, Transform, Invertible):
    r"""
    A piecewise affine transformation. This is composed of a number of
    triangles defined be a set of source and target vertices. These vertices
    are related by a common triangle list. No limitations on the nature of
    the triangle list are imposed. Points can then be mapped via
    barycentric coordinates from the source to the target space.
    Trying to map points that are not contained by any source triangle
    throws a TriangleContainmentError, which contains diagnostic information.

    Parameters
    ----------
    source : :class:`menpo.shape.PointCloud` or :class:`menpo.shape.TriMesh`
        The source points. If a TriMesh is provided, the triangulation on
        the TriMesh is used. If a :class:`menpo.shape.PointCloud`
        is provided, a Delaunay triangulation of the source is performed
        automatically.
    target : :class:`PointCloud`
        The target points. Note that the trilist is entirely decided by
        the source.

    Raises
    ------
    ValueError
        Source and target must both be 2D.

    TriangleContainmentError
        All points to apply must be contained in a source triangle. Check
        `error.points_outside_source_domain` to handle this case.
    """
    def __init__(self, source, target):
        from menpo.shape import TriMesh  # to avoid circular import
        if not isinstance(source, TriMesh):
            source = TriMesh(source.points)
        Alignment.__init__(self, source, target)
        if self.n_dims != 2:
            raise ValueError("source and target must be 2 "
                             "dimensional")
        self.ti, self.tij, self.tik = None, None, None
        self._rebuild_target_vectors()

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: int
        """
        return self.source.n_tris

    @property
    def trilist(self):
        r"""
        The triangle list.

        :type: (`n_tris`, 3) ndarray
        """
        return self.source.trilist

    def _rebuild_target_vectors(self):
        r"""
        Rebuild the vectors that are used in the apply method. This needs to
        be called whenever the target is changed.
        """
        t = self.target.points[self.trilist]
        # get vectors ij ik for the target
        self.tij, self.tik = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
        # target i'th vertex positions
        self.ti = t[:, 0]

    def _sync_state_from_target(self):
        r"""
        PWA is particularly efficient to sync from target - we don't have to
        do much at all, just rebuild the target vectors.
        """
        self._rebuild_target_vectors()

    def _apply(self, x, **kwargs):
        """
        Applies this transform to a new set of vectors.

        Parameters
        ----------
        x : (K, 2) ndarray
            Points to apply this transform to.

        Returns
        -------
        transformed : (K, 2) ndarray
            The transformed array.
        """
        tri_index, alpha, beta = self.index_alpha_beta(x)
        return (self.ti[tri_index] +
                alpha[:, None] * self.tij[tri_index] +
                beta[:, None] * self.tik[tri_index])

    @abc.abstractmethod
    def index_alpha_beta(self, points):
        """
        Finds for each input point the index of it's bounding triangle
        and the alpha and beta value for that point in the triangle. Note
        this means that the following statements will always be true:
            alpha + beta <= 1
            alpha >= 0
            beta >= 0
        for each triangle result.
        Trying to map a point that does not exist in a
        triangle throws a TriangleContainmentError.

        Parameters
        ----------
        points : (K, 2) ndarray
            Points to test.

        Returns
        -------
        tri_index : (L,) ndarray
            triangle index for each of the `points`, assigning each
            point to it's containing triangle.
        alpha : (L,) ndarray
            Alpha for containing triangle of each point.
        beta : (L,) ndarray
            Beta for containing triangle of each point.

        Raises
        ------
        TriangleContainmentError
        All `points` must be contained in a source triangle. Check
        `error.points_outside_source_domain` to handle this case.
        """
        pass

    @property
    def has_true_inverse(self):
        return True

    def pseudoinverse(self):
        from menpo.shape import PointCloud, TriMesh  # to avoid circular import
        new_source = TriMesh(self.target.points, self.source.trilist)
        new_target = PointCloud(self.source.points)
        return type(self)(new_source, new_target)


class PythonPWA(AbstractPWA):

    def __init__(self, source, target):
        super(PythonPWA, self).__init__(source, target)
        si, sij, sik = barycentric_vectors(self.source.points, self.trilist)
        self.s, self.sij, self.sik = si, sij, sik

    def index_alpha_beta(self, points):
        return index_alpha_beta(self.s, self.sij, self.sik, points)


class CachedPWA(PythonPWA):

    def __init__(self, source, target):
        super(CachedPWA, self).__init__(source, target)
        self._applied_points, self._iab = None, None

    def index_alpha_beta(self, points):
        if (self._applied_points is None or not
                np.all(points == self._applied_points)):
            self._applied_points = points
            self._iab = PythonPWA.index_alpha_beta(self, points)
        return self._iab


class CythonPWA(AbstractPWA):
    r"""
    A piecewise affine transformation.

    The apply method in this case involves dotting the triangle vectors with
    the values of alpha and beta found. The calculation of alpha and beta is
     done in C, and a hash map is used to cache lookup values.

    Parameters
    ----------
    source : :class:`menpo.shape.PointCloud` or :class:`menpo.shape.TriMesh`
        The source points. If a TriMesh is provided, the triangulation on
        the TriMesh is used. If a :class:`menpo.shape.PointCloud`
        is provided, a Delaunay triangulation of the source is performed
        automatically.
    target : :class:`PointCloud`
        The target points. Note that the trilist is entirely decided by
        the source.

    Raises
    ------
    ValueError
        Source and target must both be 2D.

    TriangleContainmentError
        All points to apply must be contained in a source triangle. Check
        `error.points_outside_source_domain` to handle this case.
    """
    def __init__(self, source, target):
        super(CythonPWA, self).__init__(source, target)
        # make sure the source and target satisfy the c requirements
        source_c = np.require(self.source.points, dtype=np.float64,
                              requirements=['C'])
        trilist_c = np.require(self.trilist, dtype=np.uint32,
                               requirements=['C'])
        # build the cython wrapped C object and store it locally
        self._fastpwa = CLookupPWA(source_c, trilist_c)

    def copy(self):
        new = Copyable.copy(self)
        new._fastpwa = deepcopy(self._fastpwa)
        return new

    def index_alpha_beta(self, points):
        points_c = np.require(points, dtype=np.float64, requirements=['C'])
        index, alpha, beta = self._fastpwa.index_alpha_beta(points_c)
        if np.any(index < 0):
            raise TriangleContainmentError(index < 0)
        else:
            return index, alpha, beta
