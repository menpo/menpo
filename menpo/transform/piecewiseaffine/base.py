import abc
import numpy as np

from menpo.base import DX, DL
from menpo.transform import Affine
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


# Note we inherit from Alignment first to get it's n_dims behavior
class AbstractPWA(Alignment, Transform, Invertible, DX, DL):
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
        -----------
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

    def _build_pseudoinverse(self):
        from menpo.shape import PointCloud, TriMesh  # to avoid circular import
        new_source = TriMesh(self.target.points, self.source.trilist)
        new_target = PointCloud(self.source.points)
        return type(self)(new_source, new_target)

    def d_dx(self, points):
        """
        Calculates the first order spatial derivative of PWA at points.

        The nature of this derivative is complicated by the piecewise nature
        of this transform. For points close to the source points of the
        transform the derivative is ill-defined. In these cases, an identity
        jacobian is returned.

        In all other cases the jacobian is equal to the containing triangle's
        d_dx.

        WARNING - presently the above behavior is only valid at the source
        points.

        Returns
        -------
        d_dx: (n_points, n_dims, n_dims) ndarray
            The first order spatial derivative of this transform

        Raises
        ------
        TriangleContainmentError:
            If any point is outside any triangle of this PWA.


        """
        # TODO check for position and return true d_dx (see docstring)
        # for the time being we assume the points are on the source landmarks
        return np.eye(2, 2)[None, ...]

    def d_dl(self, points):
        """
        Returns the jacobian of the warp at each point given in relation to the
        source points.

        Parameters
        ----------
        points : (n_points, 2) ndarray
            The points to calculate the Jacobian for.

        Returns
        -------
        d_dl : (n_points, n_centres, 2) ndarray
            The Jacobian for each of the given points over each point in
            the source points.

        """
        tri_index, alpha_i, beta_i = self.index_alpha_beta(points)
        # for the jacobian we only need
        # gamma = 1 - alpha - beta
        # for each vertex (i, j, & k)
        # gamma is the 'far edge' weighting wrt the vertex in question.
        # given gamma implicitly for the first vertex in our trilist,
        # we can permute around to get the others. (e.g. rotate CW around
        # the triangle to get the j'th vertex-as-prime variant,
        # and once again to the kth).
        #
        # alpha_j = 1 - alpha_i - beta_i
        # gamma_j = alpha_i
        # gamma_k = beta_i
        #
        # TODO this ordering is empirically correct but I don't know why..
        #
        # we stack all the gamma's together
        # so gamma_ijk.shape = (n_sample_points, 3)
        gamma_ijk = np.hstack(((1 - alpha_i - beta_i)[:, None],
                               alpha_i[:, None],
                               beta_i[:, None]))
        # the jacobian wrt source is of shape
        # (n_sample_points, n_source_points, 2)
        jac = np.zeros((points.shape[0], self.n_points, 2))
        # per sample point, find the source points for the ijk vertices of
        # the containing triangle - only these points will get a non 0
        # jacobian value
        ijk_per_point = self.trilist[tri_index]
        # to index into the jacobian, we just need a linear iterator for the
        # first axis - literally [0, 1, ... , n_sample_points]. The
        # reshape is needed to make it broadcastable with the other indexing
        # term, ijk_per_point.
        linear_iterator = np.arange(points.shape[0]).reshape((-1, 1))
        # in one line, we are done.
        jac[linear_iterator, ijk_per_point] = gamma_ijk[..., None]
        return jac


class DiscreteAffinePWA(AbstractPWA):
    r"""
    A piecewise affine transformation.

    Builds `Affine` objects for each triangle. apply involves
    finding the containing triangle for each input point, and then applying
    the appropriate Affine Transform.

    For small numbers of Triangles (order 10) this is a useful explicit
    approach that can be useful for debugging. For larger numbers of
    triangles it's use is strongly discouraged.

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
        All points to apply_inplace must be contained in a source triangle. Check
        `error.points_outside_source_domain` to handle this case.
    """
    def __init__(self, source, target):
        AbstractPWA.__init__(self, source, target)
        self._produce_affine_transforms_per_tri()

    def _produce_affine_transforms_per_tri(self):
        r"""
        Compute the affine transformation between each triangle in the source
        and target. This is calculated analytically.
        """
        # we permute the axes of the indexed point set to have shape
        # [3, n_dims, n_tris] for ease of indexing in.
        s = np.transpose(self.source.points[self.trilist],
                         axes=[1, 2, 0])
        t = np.transpose(self.target.points[self.trilist],
                         axes=[1, 2, 0])
        # sik
        # ^^^
        # ||\- the k'th point
        # ||
        # |vector between end (j or k) and i
        # source [target]
        # if i is absent, it is the position of the ijk point.
        # (not a _vector_ between points)
        # get vectors ij ik for source and target
        sij, sik = s[1] - s[0], s[2] - s[0]
        tij, tik = t[1] - t[0], t[2] - t[0]

        # source vertex positions
        si, sj, sk = s[0], s[1], s[2]
        ti = t[0]

        d = (sij[0] * sik[1]) - (sij[1] * sik[0])

        c_x = (sik[1] * tij - sij[1] * tik) / d
        c_y = (sij[0] * tik - sik[0] * tij) / d
        c_t = ti + (tij * (si[1] * sik[0] - si[0] * sik[1]) +
                    tik * (si[0] * sij[1] - si[1] * sij[0])) / d
        ht = np.repeat(np.eye(3)[..., None], self.n_tris, axis=2)
        ht[:2, 0] = c_x
        ht[:2, 1] = c_y
        ht[:2, 2] = c_t
        transforms = []
        for i in range(self.n_tris):
            transforms.append(Affine(ht[..., i]))

        # store our state out
        self.transforms = transforms
        self.s, self.t = s, t
        self.sij, self.sik = sij, sik
        self.tij, self.tik = tij, tik

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
        -----------
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
        alpha, beta = self.alpha_beta(points)
        each_point = np.arange(points.shape[0])
        index = self._containment_from_alpha_beta(alpha, beta)
        return index, alpha[each_point, index], beta[each_point, index]

    def alpha_beta(self, points):
        r"""
        Calculates the alpha and beta values (barycentric coordinates) for each
        triangle for all points provided. Note that this does not raise a
        TriangleContainmentError.

        Parameters
        ----------
        points : (K, 2) ndarray
            Points to calculate the barycentric coordinates for.

        Returns
        --------
        alpha : (K, `n_tris`)
            The alpha for each point and triangle. Alpha can be interpreted
            as the contribution of the ij vector to the position of the
            point in question.
        beta : (K, `n_tris`)
            The beta for each point and triangle. Beta can be interpreted as
             the contribution of the ik vector to the position of the point
             in question.
        """
        ip, ij, ik = (points[..., None] - self.s[0]), self.sij, self.sik
        dot_jj = np.einsum('dt, dt -> t', ij, ij)
        dot_kk = np.einsum('dt, dt -> t', ik, ik)
        dot_jk = np.einsum('dt, dt -> t', ij, ik)
        dot_pj = np.einsum('vdt, dt -> vt', ip, ij)
        dot_pk = np.einsum('vdt, dt -> vt', ip, ik)

        d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
        alpha = (dot_kk * dot_pj - dot_jk * dot_pk) * d
        beta = (dot_jj * dot_pk - dot_jk * dot_pj) * d
        return alpha, beta

    @staticmethod
    def _containment_from_alpha_beta(alpha, beta):
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
        point_containment = np.logical_and(
                            np.logical_and(alpha >= 0, beta >= 0),
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

    def _sync_state_from_target(self):
        r"""
        DiscreteAffinePWATransform is particularly inefficient to sync
        from target - we just have to manually go through and rebuild all
        the affine transforms.
        """
        self._produce_affine_transforms_per_tri()

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
        # build a list of points in each triangle for each triangle
        x_per_tri = [x[tri_index == i] for i in xrange(self.n_tris)]
        # zip the transforms and the list to apply to make the transformed x
        x_per_tri_tran = [t.apply(p) for p, t in zip(x_per_tri,
                                                     self.transforms)]
        x_transformed = np.ones_like(x) * np.nan
        # loop through each triangle, indexing into the x_transformed array
        # for points in that triangle and replacing the value of x with x_t
        for i, x_t in enumerate(x_per_tri_tran):
            x_transformed[tri_index == i] = x_t
        return x_transformed


class CachedPWA(AbstractPWA):
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
        super(CachedPWA, self).__init__(source, target)
        # make sure the source and target satisfy the c requirements
        source_c = np.require(self.source.points, dtype=np.float64,
                              requirements=['C'])
        trilist_c = np.require(self.trilist, dtype=np.uint32,
                               requirements=['C'])
        # build the cython wrapped C object and store it locally
        self._fastpwa = CLookupPWA(source_c, trilist_c)
        self.ti, self.tij, self.tik = None, None, None
        self._rebuild_target_vectors()

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

    def index_alpha_beta(self, points):
        points_c = np.require(points, dtype=np.float64, requirements=['C'])
        index, alpha, beta = self._fastpwa.index_alpha_beta(points_c)
        if np.any(index < 0):
            raise TriangleContainmentError(index < 0)
        else:
            return index, alpha, beta

    def _sync_state_from_target(self):
        r"""
        CachedPWATransform is particularly efficient to sync
        from target - we don't have to do much at all, just rebuild the target
        vectors.
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

