import numpy as np
from scipy.spatial import distance
from pybug.align.nonrigid.exceptions import TPSError
from pybug.transform.base import Transform


class TPSTransform(Transform):

    def __init__(self, tps):
        self.tps = tps
        self.n_dim = self.tps.n_dim

    def _apply(self, points, affine_free=False):
        """
        TPS transform of input x (f) and the affine-free
        TPS transform of the input x (f_affine_free)
        """
        if points.shape[1] != self.n_dim:
            raise TPSError('TPS can only be used on 2D data.')
        x = points[..., 0][:, None]
        y = points[..., 1][:, None]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_C = self.tps.coefficients[-3]
        c_affine_X = self.tps.coefficients[-2]
        c_affine_Y = self.tps.coefficients[-1]
        # the affine warp component
        f_affine = c_affine_C + c_affine_X * x + c_affine_Y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        dist = distance.cdist(self.tps.source, points)
        kernel_dist = self.tps.kernel.phi(dist)
        # grab the affine free components of the warp
        c_affine_free = self.tps.coefficients[:-3]
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
        may be constant
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dp: n_points x n_params x n_dims ndarray representing
            the Jacobian of the transform evaluated at the previous points.
        """
        pass

    # TODO: revise this function and try to speed it up!!!
    def jacobian_source(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks.
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dx_s: n_points x n_landmarks x n_dims ndarray representing
            the Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points.
        """
        # I've been tempted to rename all TPS properties so that they match
        # the names on my transfer, that would, perhaps, facilitate the
        # understanding of this method... Let me know what do you think...
        # It'd be quite a lot of renaming I guess...

        n_lms = self.tps.n_landmarks
        n_pts = points.shape[0]

        # TPS kernel (nonlinear + affine)
        dist = distance.cdist(self.tps.source, points)
        kernel_dist = self.tps.kernel.phi(dist)
        k = np.concatenate([kernel_dist, np.ones((1, n_pts)), points.T], axis=0)
        inv_L = np.linalg.inv(self.tps.L)

        dL_dx = np.zeros(self.tps.L.shape + (n_lms,))
        dL_dy = np.zeros(self.tps.L.shape + (n_lms,))
        s = self.tps.source[:, np.newaxis, :] - self.tps.source
        r = distance.squareform(distance.pdist(self.tps.source))
        r[r == 0] = 1
        aux = 2 * (1 + np.log(r**2))[..., None] * s
        dW_dx = np.zeros((n_pts, n_lms, 2))
        for i in np.arange(n_lms):
            dK_dxyi = np.zeros((self.tps.K.shape + (2,)))
            dK_dxyi[i] = aux[i]
            dK_dxyi[:, i] = -aux[:, i]

            dP_dxi = np.zeros_like(self.tps.P)
            dP_dyi = np.zeros_like(self.tps.P)
            dP_dxi[i, 1] = -1
            dP_dyi[i, 2] = -1

            dL_dx[:n_lms, :n_lms, i] = dK_dxyi[..., 0]
            dL_dx[:n_lms, n_lms:, i] = dP_dxi
            dL_dx[n_lms:, :n_lms, i] = dP_dxi.T

            dL_dy[:n_lms, :n_lms, i] = dK_dxyi[..., 1]
            dL_dy[:n_lms, n_lms:, i] = dP_dyi
            dL_dy[n_lms:, :n_lms, i] = dP_dyi.T
            # new bit
            aux3 = np.zeros((self.tps.Y.shape[1], n_pts))
            aux4 = np.zeros((self.tps.Y.shape[1], n_pts))
            aux5 = (points - self.tps.source[i, :])
            aux3[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 0]
            aux4[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 1]
            dW_dx[:, i, 0] = (self.tps.Y[0].dot(
                (-inv_L.dot(dL_dx[..., i].dot(inv_L)))).dot(k).T +
                self.tps.coefficients[:, 0].dot(aux3))
            dW_dx[:, i, 1] = (self.tps.Y[1].dot(
                (-inv_L.dot(dL_dy[..., i].dot(inv_L)))).dot(k).T +
                self.tps.coefficients[:, 1].dot(aux4))

        return dW_dx

    def jacobian_target(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the target landmarks.
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dx_t: n_points x n_landmarks x n_dims ndarray representing
            the Jacobian of the transform wrt to the target landmarks evaluated
            at the previous points.
        """
        pass

    # TODO: this is needed for composition
    def jacobian_points(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the the points to which
        the warp is applied to.
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dx:  n_points x n_dims x n_dims ndarray representing
            the Jacobian of the transform wrt the points to which the
            transform is applied to.
        """
        #Y = np.hstack([points.T, np.zeros([2, 3])])
        #coefficients = np.linalg.solve(self.tps.L, Y.T)

        abs_dist = distance.cdist(self.tps.source, self.tps.source)

        vec_dist = self.tps.source - self.tps.source[:, np.newaxis]

        for i in range(0, 68):
            vec_dist[:, i, :] = (self.tps.source[i, :] -
                              self.tps.source)

        dk_dx = np.zeros((self.tps.n_landmarks + 3,
                          self.tps.n_landmarks,
                          self.n_dim))
        aux_1 = self.tps.kernel_derivative(abs_dist)
        dk_dx[:-3, :] = aux_1[..., np.newaxis] * vec_dist

        aux_2 = np.array([[0, 0],
                          [1, 0],
                          [0, 1]])
        dk_dx[-3:, :] = aux_2[:, np.newaxis]

        return np.einsum('ij, ikl -> kjl', self.tps.coefficients, dk_dx)


    # TODO: revise this function and try to speed it up!!!
    def weight_points(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks
        assuming that he target is equal to the source. This is a special
        case of the Jacobian wrt to the source landmarks that is used in AAMs
        to weight the relative importance of each pixel in the reference
        frame wrt to each one of the source landmarks.
        :param points: n_points x n_dims ndarray representing the points at
            which the Jacobian will be evaluated.
        :return dW/dx: n_points x n_landmarks x n_dims ndarray representing
            the Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points and assuming that the target is equal to
            the source.
        """
        n_lms = self.tps.n_landmarks
        n_pts = points.shape[0]

        # TPS kernel (nonlinear + affine)
        dist = distance.cdist(self.tps.source, points)
        kernel_dist = self.tps.kernel.phi(dist)
        k = np.concatenate([kernel_dist, np.ones((1, n_pts)), points.T], axis=0)
        inv_L = np.linalg.inv(self.tps.L)

        dL_dx = np.zeros(self.tps.L.shape + (n_lms,))
        dL_dy = np.zeros(self.tps.L.shape + (n_lms,))
        s = self.tps.source[:, np.newaxis, :] - self.tps.source
        r = distance.squareform(distance.pdist(self.tps.source))
        r[r == 0] = 1
        aux = 2 * (1 + np.log(r**2))[..., None] * s
        dW_dx = np.zeros((n_pts, n_lms, 2))

        pseudo_target = np.hstack([self.tps.source.T, np.zeros([2, 3])])

        for i in np.arange(n_lms):
            dK_dxyi = np.zeros((self.tps.K.shape + (2,)))
            dK_dxyi[i] = aux[i]
            dK_dxyi[:, i] = -aux[:, i]

            dP_dxi = np.zeros_like(self.tps.P)
            dP_dyi = np.zeros_like(self.tps.P)
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
        Composes two transforms together: W(x;p) <- W(x;p) o W(x;delta_p)
        :param a: transform of the same type as this object
        """
        pass

    def inverse(self):
        """
        Returns the inverse of the transform, if applicable
        :raise NonInvertable if transform has no inverse
        """
        pass

    @property
    def n_parameters(self):
        """
        There is a parameter for each dimension, and thus two parameters per
        landmark + the parameters of a 2D affine transform:
        (2 * num_landmarks) + 6
        :return:
        """
        return (2 * self.tps.n_landmarks) + 6

    def as_vector(self):
        """
        Return the parameters of the transform as a 1D ndarray
        """
        pass

    def from_vector(self, vectorized_instance):
        """
        Return the parameters of the transform as a 1D ndarray
        """
        pass
