import numpy as np
from scipy.spatial import distance
from pybug.shape import PointCloud
from pybug.transform.base import PureAlignmentTransform
from pybug.basis.rbf import R2LogR2


class TPS(PureAlignmentTransform):
    r"""
    The thin plate splines (TPS) alignment between 2D source and target
    landmarks.

    ``kernel`` can be used to specify an alternative kernel function. If
    ``None`` is supplied, the ``r**2 log(r**2)`` kernel will be used.

    Parameters
    ----------
    source : (N, 2) ndarray
        The source points to apply the tps from
    target : (N, 2) ndarray
        The target points to apply the tps to
    kernel : func, optional
        The kernel function to apply.

        Default: ``r**2 log(r**2)``

    Raises
    ------
    ValueError
        TPS is only supported on 2-dimensional data
    """

    def __init__(self, source, target, kernel=None):
        super(TPS, self).__init__(source, target)
        if self.n_dims != 2:
            raise ValueError('TPS can only be used on 2D data.')
        if kernel is None:
            kernel = R2LogR2()
        self.kernel = kernel
        self.pairwise_norms = self.source.distance_to(self.source)
        self.k = self.kernel.phi(self.pairwise_norms)
        self.p = np.concatenate(
            [np.ones([self.n_points, 1]), self.source.points], axis=1)
        o = np.zeros([3, 3])
        top_l = np.concatenate([self.k, self.p], axis=1)
        bot_l = np.concatenate([self.p.T, o], axis=1)
        self.l = np.concatenate([top_l, bot_l], axis=0)
        self.v, self.y, self.coefficients = None, None, None
        self._build_coefficients()

    def _build_coefficients(self):
        self.v = self.target.points.T.copy()
        self.y = np.hstack([self.v, np.zeros([2, 3])])
        self.coefficients = np.linalg.solve(self.l, self.y.T)

    @property
    def n_parameters(self):
        """

        :type: int
        """
        raise NotImplementedError("n_parameters for TPS needs to be "
                                  "implemented")

    @property
    def has_true_inverse(self):
        return False

    def _build_pseudoinverse(self):
        return TPS(self.target, self.source, kernel=self.kernel)

    def _target_setter(self, new_target):
        self._target = new_target
        # now the target is updated, we only have to rebuild the
        # coefficients.
        self._build_coefficients()

    def _apply(self, points):
        """
        Performs a TPS transform on the given points.

        Parameters
        ----------
        points : (N, D) ndarray
            The points to transform.
        affine_free : bool, optional
            If ``True`` the affine free component is also returned separately.

            Default: ``False``

        Returns
        --------
        f : (N, D) ndarray
            The transformed points
        f_affine_free : (N, D) ndarray
            The transformed points without the affine components applied.
        """
        points = PointCloud(points)
        if points.n_dims != self.n_dims:
            raise ValueError('TPS can only be applied to 2D data.')
        x = points.points[..., 0][:, None]
        y = points.points[..., 1][:, None]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_c = self.coefficients[-3]
        c_affine_x = self.coefficients[-2]
        c_affine_y = self.coefficients[-1]
        # the affine warp component
        f_affine = c_affine_c + c_affine_x * x + c_affine_y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        dist = self.source.distance_to(points)
        kernel_dist = self.kernel.phi(dist)
        # grab the affine free components of the warp
        c_affine_free = self.coefficients[:-3]
        # build the affine free warp component
        f_affine_free = np.sum(c_affine_free[:, None, :] *
                               kernel_dist[..., None],
                               axis=0)
        return f_affine + f_affine_free

    def jacobian(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the parameters.

        This isn't implemented yet, and can't be implemented until
        n_parameters is fixed and a suitable parametrisation chosen for TPS.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform evaluated at the previous points.
        """
        raise NotImplementedError("n_parameters for TPS needs to be "
                                  "implemented")

    def jacobian_points(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the the points to which
        the warp is applied to.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform wrt the points to which the
            transform is applied to.
        """
        vec_dist = np.subtract(self.source.points[:, None],
                               self.source.points)

        dk_dx = np.zeros((self.n_points + 3,
                          self.n_points,
                          self.n_dims))
        kernel_derivative = (self.kernel.derivative(self.pairwise_norms) /
                             self.pairwise_norms)
        dk_dx[:-3, :] = kernel_derivative[..., None] * vec_dist

        affine_derivative = np.array([[0, 0],
                                     [1, 0],
                                     [0, 1]])
        dk_dx[-3:, :] = affine_derivative[:, np.newaxis]

        return np.einsum('ij, ikl -> klj', self.coefficients, dk_dx)

    # TODO: revise me
    def jacobian_source(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points.
        """
        points_pc = PointCloud(points)
        n_lms = self.n_points
        n_pts = points_pc.n_points

        # TPS kernel (nonlinear + affine)
        dist = self.source.distance_to(points_pc)
        kernel_dist = self.kernel.phi(dist)
        k = np.concatenate([kernel_dist, np.ones((1, n_pts)),
                            points.T],
                           axis=0)
        inv_L = np.linalg.inv(self.l)

        dL_dx = np.zeros(self.l.shape + (n_lms,))
        dL_dy = np.zeros(self.l.shape + (n_lms,))
        s = self.source.points[:, np.newaxis, :] - self.source.points
        r = distance.squareform(distance.pdist(self.source.points))
        r[r == 0] = 1
        aux = 2 * (1 + np.log(r**2))[..., None] * s
        dW_dx = np.zeros((n_pts, n_lms, 2))

        # Fix log(0)
        dist[dist == 0] = 1
        for i in np.arange(n_lms):
            dK_dxyi = np.zeros((self.k.shape + (2,)))
            dK_dxyi[i] = aux[i]
            dK_dxyi[:, i] = -aux[:, i]

            dP_dxi = np.zeros_like(self.p)
            dP_dyi = np.zeros_like(self.p)
            dP_dxi[i, 1] = -1
            dP_dyi[i, 2] = -1

            dL_dx[:n_lms, :n_lms, i] = dK_dxyi[..., 0]
            dL_dx[:n_lms, n_lms:, i] = dP_dxi
            dL_dx[n_lms:, :n_lms, i] = dP_dxi.T

            dL_dy[:n_lms, :n_lms, i] = dK_dxyi[..., 1]
            dL_dy[:n_lms, n_lms:, i] = dP_dyi
            dL_dy[n_lms:, :n_lms, i] = dP_dyi.T
            # new bit
            aux3 = np.zeros((self.y.shape[1], n_pts))
            aux4 = np.zeros((self.y.shape[1], n_pts))
            aux5 = (points - self.source.points[i, :])
            # TODO this is hardcoded and should be set based on kernel
            aux3[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 0]
            aux4[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 1]
            dW_dx[:, i, 0] = (self.y[0].dot(
                (-inv_L.dot(dL_dx[..., i].dot(inv_L)))).dot(k).T +
                self.coefficients[:, 0].dot(aux3))
            dW_dx[:, i, 1] = (self.y[1].dot(
                (-inv_L.dot(dL_dy[..., i].dot(inv_L)))).dot(k).T +
                self.coefficients[:, 1].dot(aux4))

        return dW_dx

    # TODO: revise this function and try to speed it up!!!
    def weight_points(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks
        assuming that he target is equal to the source. This is a special
        case of the Jacobian wrt to the source landmarks that is used in AAMs
        to weight the relative importance of each pixel in the reference
        frame wrt to each one of the source landmarks.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points and assuming that the target is equal to
            the source.
        """
        points_pc = PointCloud(points)
        n_lms = self.n_points
        n_pts = points.n_points

        # TPS kernel (nonlinear + affine)
        dist = self.source.distance_to(points_pc)
        kernel_dist = self.kernel.phi(dist)
        k = np.concatenate([kernel_dist, np.ones((1, n_pts)), points.T], axis=0)
        inv_L = np.linalg.inv(self.l)

        dL_dx = np.zeros(self.l.shape + (n_lms,))
        dL_dy = np.zeros(self.l.shape + (n_lms,))
        s = self.source.points[:, np.newaxis, :] - self.source.points
        r = distance.squareform(distance.pdist(self.source.points))
        r[r == 0] = 1
        aux = 2 * (1 + np.log(r**2))[..., None] * s
        dW_dx = np.zeros((n_pts, n_lms, 2))

        pseudo_target = np.hstack([self.source.points.T, np.zeros([2, 3])])

        for i in np.arange(n_lms):
            dK_dxyi = np.zeros((self.k.shape + (2,)))
            dK_dxyi[i] = aux[i]
            dK_dxyi[:, i] = -aux[:, i]

            dP_dxi = np.zeros_like(self.p)
            dP_dyi = np.zeros_like(self.p)
            dP_dxi[i, 1] = -1
            dP_dyi[i, 2] = -1

            dL_dx[:n_lms, :n_lms, i] = dK_dxyi[..., 0]
            dL_dx[:n_lms, n_lms:, i] = dP_dxi
            dL_dx[n_lms:, :n_lms, i] = dP_dxi.T

            dL_dy[:n_lms, :n_lms, i] = dK_dxyi[..., 1]
            dL_dy[:n_lms, n_lms:, i] = dP_dyi
            dL_dy[n_lms:, :n_lms, i] = dP_dyi.T

            dW_dx[:, i, 0] = (pseudo_target[0].dot(
                (-inv_L.dot(dL_dx[..., i].dot(inv_L)))).dot(k).T)
            dW_dx[:, i, 1] = (pseudo_target[1].dot(
                (-inv_L.dot(dL_dy[..., i].dot(inv_L)))).dot(k).T)

        return dW_dx

    def as_vector(self):
        raise NotImplementedError("TPS as_vector is not implemented yet.")

    def from_vector(self, flattened):
        raise NotImplementedError("TPS from_vector is not implemented yet.")

    def from_vector_inplace(self, vector):
        pass
