import numpy as np
from scipy.spatial import distance
from pybug.shape import PointCloud
from pybug.transform.base import AlignmentTransform
from pybug.basis.rbf import R2LogR2


class TPS(AlignmentTransform):
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
        self.v = self.target.points.T.copy()
        self.y = np.hstack([self.v, np.zeros([2, 3])])
        self.pairwise_norms = self.source.distance_to(self.source)
        if kernel is None:
            kernel = R2LogR2()
        self.kernel = kernel
        self.k = self.kernel.phi(self.pairwise_norms)
        self.p = np.concatenate(
            [np.ones([self.n_points, 1]), self.source.points], axis=1)
        o = np.zeros([3, 3])
        top_l = np.concatenate([self.k, self.p], axis=1)
        bot_l = np.concatenate([self.p.T, o], axis=1)
        self.l = np.concatenate([top_l, bot_l], axis=0)
        self.coefficients = np.linalg.solve(self.l, self.y.T)

    def _apply(self, points, affine_free=False):
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
        if affine_free:
            return f_affine + f_affine_free, f_affine_free
        else:
            return f_affine + f_affine_free

    def jacobian(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the parameters - this
        may be constant.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform evaluated at the previous points.
        """
        pass

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
            aux5 = (points - self.source[i, :])
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

    def jacobian_target(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the target landmarks.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dp : (N, P, D) ndarray
            The Jacobian of the transform wrt to the target landmarks evaluated
            at the previous points.
        """
        raise NotImplementedError("TPS jacobian_target is not implemented "
                                  "yet.")

    # TODO: this is needed for composition
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

    def compose(self, a):
        """
        Composes two transforms together::

            ``W(x;p) <- W(x;p) o W(x;delta_p)``

        Parameters
        ----------
        a : :class:`TPSTransform`
            TPS transform to compose with.

        Returns
        -------
        composed : :class:`TPSTransform`
            The result of the composition.
        """
        raise NotImplementedError("TPS compose is not implemented yet.")

    def inverse(self):
        """
        Returns the inverse of the transform, if applicable.

        Returns
        -------
        inverse : :class:`TPSTransform`
            The inverse of the transform.
        """
        raise NotImplementedError("TPS inverse is not implemented yet.")

    @property
    def n_parameters(self):
        """
        Number of parameters: ``(2 * n_landmarks) + 6``.

        :type: int

        There is a parameter for each dimension, and thus two parameters per
        landmark + the parameters of a 2D affine transform
        ``(2 * n_landmarks) + 6``
        """
        return (2 * self.n_points) + 6

    def as_vector(self):
        raise NotImplementedError("TPS as_vector is not implemented yet.")

    def from_vector(self, flattened):
        raise NotImplementedError("TPS from_vector is not implemented yet.")

    def _update_from_target(self, new_target):
        pass
