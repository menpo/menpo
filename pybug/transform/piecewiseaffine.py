import numpy as np
from pybug.exceptions import DimensionalityError
from pybug.shape import TriMesh
from pybug.transform import AffineTransform, Transform


class PiecewiseAffineTransform(Transform):
    r"""
    A piecewise affine transformation. This is composed of a number of
    triangles defined be a set of source and target vertices. These vertices
    are related by a common triangle list. Points can then be mapped via
    barycentric coordinates from the source to the target space. Points within
    related triangles can be transformed via the same affine transform. Points
    outside of the convex hull of the source points have an undefined
    transform.

    Parameters
    ----------
    source : (N, 2) ndarray
        The source points.
    target : (N, 2) ndarray
        The target points.
    trilist : (M, 3) ndarray
        A common triangulation for the ``source`` and ``target``.

    Raises
    ------
    DimensionalityError
        Source and target must have the same dimensionality.

        Source and target must be 2D.
    """

    def __init__(self, source, target, trilist):
        self.source = TriMesh(source, trilist)
        self.target = TriMesh(target, trilist)
        self.n_dims = self.source.n_dims
        if self.source.n_dims != self.target.n_dims:
            raise DimensionalityError("source and target must have the same "
                                      "dimension")
        if self.source.n_dims != 2:
            raise DimensionalityError("source and target must be 2 "
                                      "dimensional")
        self._produce_affine_transforms_per_tri()

    @property
    def n_tris(self):
        r"""
        The number of triangles in the triangle list.

        :type: int
        """
        return self.source.n_tris

    @property
    def n_points(self):
        r"""
        The number of points in the source and target.

        :type: int
        """
        return self.source.n_points

    @property
    def trilist(self):
        r"""
        The triangle list.

        :type: (``n_tris``, 3) ndarray
        """
        return self.source.trilist

    def _produce_affine_transforms_per_tri(self):
        r"""
        Compute the affine transformation between each triangle in the source
        and target. This is calculated analytically.
        """
        # we permute the axes of the indexed point set to have shape
        # [3, n_dims, n_tris] for ease of indexing in.
        s = np.transpose(self.source.points[self.source.trilist],
                         axes=[1, 2, 0])
        t = np.transpose(self.target.points[self.target.trilist],
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
        for i in range(self.source.n_tris):
            transforms.append(AffineTransform(ht[..., i]))

        # store our state out
        self.transforms = transforms
        self.s, self.t = s, t
        self.sij, self.sik = sij, sik
        self.tij, self.tik = tij, tik

    def alpha_beta(self, points):
        """
        Calculates the alpha and beta values (barycentric coordinates) for each
        triangle for all points provided.

        Parameters
        ----------
        points : (K, 2) ndarray
            Points to calculate the barycentric coordinates for.

        Returns
        --------
        alpha : (K, ``n_tris``)
            The alpha for each point and triangle.
        beta : (K, ``n_tris``)
            The beta for each point and triangle.
        """
        ip, ij, ik = (points[..., None] - self.s[0]), self.sij, self.sik
        # todo this could be cached if tri_containment is being tested at
        # many points
        dot_jj = np.einsum('dt, dt -> t', ij, ij)
        dot_kk = np.einsum('dt, dt -> t', ik, ik)
        dot_jk = np.einsum('dt, dt -> t', ij, ik)
        dot_pj = np.einsum('vdt, dt -> vt', ip, ij)
        dot_pk = np.einsum('vdt, dt -> vt', ip, ik)

        d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
        alpha = (dot_jj * dot_pk - dot_jk * dot_pj) * d
        beta = (dot_kk * dot_pj - dot_jk * dot_pk) * d
        return alpha, beta

    def _containment_from_alpha_beta(self, alpha, beta):
        r"""
        Check ``alpha`` and ``beta`` are within a triangle (``alpha >= 0``,
        ``beta >= 0``, ``alpha + beta <= 1``). Returns the indices of the
        triangles that are ``alpha`` and ``beta`` are in.

        Parameters
        ----------
        alpha: (K, ``n_tris``) ndarray
            Alpha for each point and triangle being tested.
        beta: (K, ``n_tris``) ndarray
            Beta for each point and triangle being tested.

        Returns
        -------
        points_in_tris : (L,) ndarray
            Index into points, such that ``points[points_in_tris]`` yields only
            points that are contained within a triangle.
        tri_index : (L,) ndarray
            triangle index for each ``points[points_in_tris]``, assigning each
            point in a triangle to the triangle index.
        """
        return np.nonzero(np.logical_and(
            np.logical_and(alpha >= 0, beta >= 0), alpha + beta <= 1))

    def tri_containment(self, points):
        """
        Finds for each input point whether it is contained in a triangle,
        and if so what triangle index it is in.

        Parameters
        -----------
        points : (K, 2) ndarray
            Points to test.

        Returns
        -------
        points_in_tris : (L,) ndarray
            Index into points, such that ``points[points_in_tris]`` yields only
            points that are contained within a triangle.
        tri_index : (L,) ndarray
            triangle index for each ``points[points_in_tris]``, assigning each
            point in a triangle to the triangle index.
        """
        alpha, beta = self.alpha_beta(points)
        return self._containment_from_alpha_beta(alpha, beta)

    def weight_points(self, points):
        """
        Returns the jacobian of the warp at each point given in relation to the
        source points.

        Parameters
        ----------
        points : (K, 2) ndarray
            The points to calculate the Jacobian for.

        Returns
        -------
        jacobian : (K, ``n_points``, 2) ndarray
            The Jacobian for each of the ``K`` given points over each point in
            the source points.
        """
        alpha_i, beta_i = self.alpha_beta(points)
        # given alpha beta implicitly for the first vertex in our trilist,
        # we can permute around to get the others. (e.g. rotate CW around
        # the triangle to get the j'th vertex-as-prime varient,
        # and once again to the kth).
        #
        # alpha_j = 1 - alpha_i - beta_i
        # beta_j = alpha_i
        #
        # alpha_k = beta_i
        # beta_k = 1 - alpha_i - beta_i
        #
        # for the jacobian we only need 1 - a - b for each vertex (i, j, & k)
        # gamma_i = 1 - alpha_i - beta_i
        # gamma_j = 1 - (1 - alpha_i - beta_i) - alpha_i = beta_i
        # gamma_k = 1 - (beta_i) - (1 - alpha_i - beta_i) = alpha_i
        # skipping the working out:
        gamma_i = 1 - alpha_i - beta_i
        gamma_j = beta_i
        gamma_k = alpha_i
        # the jacobian wrt source is of shape (n_points, n_source lm, n_dim)
        jac = np.zeros((points.shape[0], self.n_points, 2))
        # now its a case of writing the gamma values in for the correct
        # triangles
        points_in_tris, tri_index = self._containment_from_alpha_beta(
            alpha_i, beta_i)
        for i in xrange(self.n_tris):
            # e.g. [0,1,2]
            points_index_in_tri_i = points_in_tris[tri_index == i]
            # e.g [3,4,7]
            vertices_index_of_tri_i = self.trilist[i]
            jac[points_index_in_tri_i, vertices_index_of_tri_i[0], :] = (
                gamma_i[points_index_in_tri_i, i][..., None])
            jac[points_index_in_tri_i, vertices_index_of_tri_i[1], :] = (
                gamma_j[points_index_in_tri_i, i][..., None])
            jac[points_index_in_tri_i, vertices_index_of_tri_i[2], :] = (
                gamma_k[points_index_in_tri_i, i][..., None])
        return jac

    def jacobian_points(self, points):
        """
        Calculates the Jacobian of the PWA warp with respect to the the points
        to which the warp is applied to. Expected to return a
        ``(n_points, n_dims, n_dims)`` shaped array, so the result is tiled
        as necessary.

        The derivative of a piecewise affine warp with respect to the points
        is simply the identity matrix for every point in the warp.

        Parameters
        ----------
        points: (N, D) ndarray
            The points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dx: (N, D, D) ndarray
            The Jacobian of the transform with respect to the points to which
            the transform is applied to.
        """
        return np.tile(np.eye(2, 2), [self.n_points, 1, 1])

    def _tri_containment_loop(self, points):
        """
        Performs the same operation as tri_containment but in C style.
        Useful as a reference for how to convert C/Matlab style to numpy
        (especially the use of einsum).
        """
        all_ij, all_ik = self.sij.T, self.sik.T
        all_i = self.s[0].T
        output = np.zeros((points.shape[0], self.n_tris), dtype=np.bool)
        for i, p in enumerate(points):
            for t in range(self.n_tris):
                ip = p - all_i[t]
                ij = all_ij[t]
                ik = all_ik[t]
                dot_jj = np.dot(ij, ij)
                dot_kk = np.dot(ik, ik)
                dot_jk = np.dot(ij, ik)
                dot_pj = np.dot(ip, ij)
                dot_pk = np.dot(ip, ik)

                d = 1.0/(dot_jj * dot_kk - dot_jk * dot_jk)
                u = (dot_jj * dot_pk - dot_jk * dot_pj) * d
                v = (dot_kk * dot_pj - dot_jk * dot_pk) * d
                output[i, t] = (u >= 0 and v >= 0) and u + v <= 1
        return np.nonzero(output)

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
        x_in_tris, tri_index = self.tri_containment(x)
        # build a list of points in each triangle for each triangle
        x_per_tri = [x[x_in_tris[tri_index == i]] for i in xrange(self.n_tris)]
        # zip the transforms and the list to apply to make the transformed x
        x_per_tri_tran = [t.apply(p) for p, t in zip(x_per_tri,
                                                     self.transforms)]
        x_transformed = np.ones_like(x) * np.nan
        # loop through each triangle, indexing into the x_transformed array
        # for points in that triangle and replacing the value of x with x_t
        for i, x_t in enumerate(x_per_tri_tran):
            x_transformed[x_in_tris[tri_index == i]] = x_t
        return x_transformed

    @property
    def n_parameters(self):
        """
        Number of parameters: ``n_tris * 6``.

        :type: int

        There is a 2D affine transformation per triangle, therefore, there are
        number of triangles * parameters for 2D affine transform number of
        parameters for a PieceWiseAffine transform: ``n_tris * 6``.
        """
        return self.n_tris * 6

    def as_vector(self):
        raise NotImplementedError("PWA as_vector is not implemented yet.")

    def compose(self, a):
        raise NotImplementedError("PWA compose is not implemented yet.")

    def from_vector(self, flattened):
        raise NotImplementedError("PWA from_vector is not implemented yet.")

    def inverse(self):
        raise NotImplementedError("PWA inverse is not implemented yet.")

    def jacobian(self, shape):
        raise NotImplementedError("PWA jacobian is not implemented yet.")
