import numpy as np
from scipy.spatial import distance
from menpo.basis.rbf import R2LogR2

from .base import Transform, Alignment, Invertible


# Note we inherit from Alignment first to get it's n_dims behavior
class ThinPlateSplines(Alignment, Transform, Invertible):
    r"""
    The thin plate splines (TPS) alignment between 2D source and target
    landmarks.

    `kernel` can be used to specify an alternative kernel function. If
    `None` is supplied, the :class:`menpo.basis.rbf.R2LogR2` kernel will be
    used.

    Parameters
    ----------
    source : (N, 2) ndarray
        The source points to apply the tps from
    target : (N, 2) ndarray
        The target points to apply the tps to
    kernel : :class:`menpo.basis.rbf.BasisFunction`, optional
        The kernel to apply.

        Default: :class:`menpo.basis.rbf.R2LogR2`

    Raises
    ------
    ValueError
        TPS is only with on 2-dimensional data
    """

    def __init__(self, source, target, kernel=None):
        Alignment.__init__(self, source, target)
        if self.n_dims != 2:
            raise ValueError('TPS can only be used on 2D data.')
        if kernel is None:
            kernel = R2LogR2(source.points)
        self.kernel = kernel
        self.k = self.kernel.apply(self.source.points)
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

    def _sync_state_from_target(self):
        # now the target is updated, we only have to rebuild the
        # coefficients.
        self._build_coefficients()

    def _apply(self, points, **kwargs):
        """
        Performs a TPS transform on the given points.

        Parameters
        ----------
        points : (N, D) ndarray
            The points to transform.

        Returns
        --------
        f : (N, D) ndarray
            The transformed points
        """
        if points.shape[1] != self.n_dims:
            raise ValueError('TPS can only be applied to 2D data.')
        x = points[..., 0][:, None]
        y = points[..., 1][:, None]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_c = self.coefficients[-3]
        c_affine_x = self.coefficients[-2]
        c_affine_y = self.coefficients[-1]
        # the affine warp component
        f_affine = c_affine_c + c_affine_x * x + c_affine_y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        kernel_dist = self.kernel.apply(points)
        # grab the affine free components of the warp
        c_affine_free = self.coefficients[:-3]
        # build the affine free warp component
        f_affine_free = kernel_dist.dot(c_affine_free)
        return f_affine + f_affine_free

    @property
    def has_true_inverse(self):
        return False

    def _build_pseudoinverse(self):
        return ThinPlateSplines(self.target, self.source, kernel=self.kernel)

    def jacobian_points(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the coordinate system.

        Parameters
        ----------
        points : (N, D)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dx : (N, D, D) ndarray
            The Jacobian of the transform wrt the coordinate system in
            which the transform is applied. Axis 0: points, Axis 1: direction
            of derivative (x or y) Axis 2: Component in which we are
            evaluating derivative (x or y)

            e.g. [7, 0, 1] = derivative wrt x on the y coordinate of the
            8th point.
        """
        dk_dx = np.zeros((points.shape[0] + 3,   # i
                          self.source.n_points,  # k
                          self.source.n_dims))   # l
        dk_dx[:-3, :] = self.kernel.jacobian_points(points)

        affine_derivative = np.array([[0, 0],
                                      [1, 0],
                                      [0, 1]])
        dk_dx[-3:, :] = affine_derivative[:, None]

        return np.einsum('ij, ikl -> klj', self.coefficients, dk_dx)

    # TODO: revise me
    def jacobian_source(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmark
        position.

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
        from menpo.shape import PointCloud
        points_pc = PointCloud(points)
        n_lms = self.n_points
        n_pts = points_pc.n_points

        kernel_dist = self.kernel.apply(points)
        k = np.concatenate([kernel_dist, np.ones([n_pts, 1]), points], axis=1)
        inv_L = np.linalg.inv(self.l)

        dL_dx = np.zeros(self.l.shape + (n_lms,))
        dL_dy = np.zeros(self.l.shape + (n_lms,))
        aux = self.kernel.jacobian_points(self.source.points)
        dW_dx = np.zeros((n_pts, n_lms, 2))

        # Fix log(0)
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
            aux3 = np.zeros((n_pts, self.y.shape[1], 2))
            aux3[:, i, :] = self.kernel.jacobian_points(points)[:, i, :]
            omega_x = -inv_L.dot(dL_dx[..., i].dot(inv_L))
            dW_dx[:, i, 0] = (k.dot(omega_x).dot(self.y[0]) +
                              aux3[..., 0].dot(self.coefficients[:, 0]))
            omega_y = -inv_L.dot(dL_dy[..., i].dot(inv_L))
            dW_dx[:, i, 1] = (k.dot(omega_y).dot(self.y[1]) +
                              aux3[..., 1].dot(self.coefficients[:, 1]))

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
        from menpo.shape import PointCloud
        points_pc = PointCloud(points)
        n_lms = self.n_points
        n_pts = points_pc.n_points

        # TPS kernel (nonlinear + affine)
        kernel_dist = self.kernel.apply(points)
        k = np.concatenate([kernel_dist, np.ones([n_pts, 1]), points], axis=1)
        inv_L = np.linalg.inv(self.l)

        dL_dx = np.zeros(self.l.shape + (n_lms,))
        dL_dy = np.zeros(self.l.shape + (n_lms,))
        s = self.source.points[:, np.newaxis, :] - self.source.points
        r = distance.squareform(distance.pdist(self.source.points))
        r[r == 0] = 1
        aux = 2 * (1 + np.log(r ** 2))[..., None] * s
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

            omega_x = -inv_L.dot(dL_dx[..., i].dot(inv_L))
            dW_dx[:, i, 0] = k.dot(omega_x).dot(pseudo_target[0])
            omega_y = -inv_L.dot(dL_dy[..., i].dot(inv_L))
            dW_dx[:, i, 1] = k.dot(omega_y).dot(pseudo_target[1])

        return dW_dx
