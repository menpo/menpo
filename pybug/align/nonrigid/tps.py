import numpy as np
from scipy.spatial import distance
from pybug.align.nonrigid.exceptions import TPSError
from pybug.align.nonrigid.base import NonRigidAlignment
from pybug.exceptions import DimensionalityError
from pybug.transform.base import Transform
from pybug.align.base import MultipleAlignment


class TPS(NonRigidAlignment):

    def __init__(self, source, target, kernel=None):
        """
        The TPS alignment between 2D source and target landmarks. kernel can
        be used to specify an alternative kernel function - if None is
        supplied, the r**2 log(r**2) kernel will be used.
        :param source:
        :param target:
        :raise:
        """
        super(TPS, self).__init__(source, target)
        if self.n_dim != 2:
            raise DimensionalityError('TPS can only be used on 2D data.')
        self.V = self.target.T.copy()
        self.Y = np.hstack([self.V, np.zeros([2, 3])])
        pairwise_norms = distance.squareform(distance.pdist(self.source))
        if kernel is None:
            kernel = r_2_log_r_2_kernel
        self.K = kernel(pairwise_norms)
        self.P = np.concatenate(
            [np.ones([self.n_landmarks, 1]), self.source], axis=1)
        O = np.zeros([3, 3])
        top_L = np.concatenate([self.K, self.P], axis=1)
        bot_L = np.concatenate([self.P.T, O], axis=1)
        self.L = np.concatenate([top_L, bot_L], axis=0)
        self.coefficients = np.linalg.solve(self.L, self.Y.T)
        self._transform_object = TPSTransform(self)

    @property
    def transform(self):
        return self._transform_object

    def view(self):
        self._view_2d()


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
        kernel_dist = r_2_log_r_2_kernel(dist)
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

    def jacobian(self, shape):
        """
        Calculates the Jacobian of the TPS warp wrt to the parameters - this
        may be constant
        :param shape
        """
        pass

    def jacobian_source(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt the source landmarks and
        evaluated at points - this may be constant.
        :param points
        """
        # I've been tempted to rename all TPS properties so that they match
        # the names on my transfer, that would, perhaps, facilitate the
        # understanding of this method... Let me know what do you think...
        # It'd be quite a lot of renaming I guess...

        # This is probably the shittest code ever written in python but I
        # just want to get this coded up as fast as possible, we can tidy it
        # up later on...

        # TPS kernel (nonlinear + affine)
        dist = distance.cdist(self.tps.source, points)
        kernel_dist = r_2_log_r_2_kernel(dist)
        k = np.concatenate([kernel_dist, np.ones((1, points.shape[0])),
                            points.T], axis=0)
        inv_L = np.linalg.inv(self.tps.L)

        # This is the tricky bit, surely this can be coded in a much better
        # way... But I just wanted to focus on getting the semantics right!
        dL_dx = np.zeros(self.tps.L.shape + (self.tps.source.shape[0],))
        dL_dy = np.zeros(self.tps.L.shape + (self.tps.source.shape[0],))
        s = np.zeros(self.tps.K.shape + (2,))
        for i in np.arange(0, self.tps.K.shape[0]):
            s[:, i, :] = self.tps.source - self.tps.source[i, :]
        r = distance.squareform(distance.pdist(self.tps.source))
        mask = r == 0
        r[mask] = 1
        aux = 2 * (1 + np.log(r**2)) * s[:, :, 0]
        aux2 = 2 * (1 + np.log(r**2)) * s[:, :, 1]
        dW_dx = np.zeros((points.shape[0], self.tps.source.shape[0], 2))
        for i in np.arange(0, self.tps.K.shape[0]):
            dK_dxi = np.zeros_like(self.tps.K)
            dK_dyi = np.zeros_like(self.tps.K)
            dK_dxi[i, :] = aux[i, :]
            dK_dxi[:, i] = -aux[:, i]
            dK_dyi[i, :] = aux2[i, :]
            dK_dyi[:, i] = -aux2[:, i]
            dP_dxi = np.zeros_like(self.tps.P)
            dP_dyi = np.zeros_like(self.tps.P)
            dP_dxi[i, 1] = -1
            dP_dyi[i, 2] = -1
            dL_dx[0:self.tps.K.shape[0], 0:self.tps.K.shape[0], i] = dK_dxi
            dL_dx[0:self.tps.K.shape[1], self.tps.K.shape[0]:, i] = dP_dxi
            dL_dx[self.tps.K.shape[0]:, 0:self.tps.K.shape[1],
            i] = dP_dxi.T
            dL_dy[0:self.tps.K.shape[0], 0:self.tps.K.shape[0], i] = dK_dyi
            dL_dy[0:self.tps.K.shape[1], self.tps.K.shape[0]:, i] = dP_dyi
            dL_dy[self.tps.K.shape[0]:, 0:self.tps.K.shape[1],
            i] = dP_dyi.T
            # new bit
            aux3 = np.zeros((self.tps.Y.shape[1], points.shape[0]))
            aux4 = np.zeros((self.tps.Y.shape[1], points.shape[0]))
            aux5 = (points - self.tps.source[i, :])
            aux3[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 0]
            aux4[i, :] = 2 * (1 + np.log(dist[i, :]**2)) * aux5[:, 1]
            dW_dx[:, i, 0] = self.tps.Y[0, :].dot((-inv_L.dot(dL_dx[:, :,
                                                     i].dot(inv_L)))).dot(k)\
                .T + self.tps.coefficients[:, 0].dot(aux3)

            dW_dx[:, i, 1] = self.tps.Y[1, :].dot((-inv_L.dot(dL_dy[:, :,
                                                     i].dot(inv_L)))).dot(k)\
                .T + self.tps.coefficients[:, 1].dot(aux4)

        return dW_dx

    def jacobian_target(self, shape):
        """
        Calculates the Jacobian of the tps warp wrt the source landmarks -
        this may be constant.
        :param shape
        """
        pass

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

    def parameters(self):
        """
        Return the parameters of the transform as a 1D ndarray
        """
        pass


# TODO: This may end up being a method in class later on ...
def r_2_log_r_2_kernel(r):
    """
    Radial basis function for TPS.
    """
    mask = r == 0
    r[mask] = 1
    U = r ** 2 * (np.log(r ** 2))
    # reset singularities to 0
    U[mask] = 0
    return U


# TODO: This may end up being a method in class later on ...
def r_2_log_r_2_kernel_derivative(r):
    """
    Derivative of the radial basis function for TPS.
    """
    mask = r == 0
    r[mask] = 1
    dUdr = 2 * r * (1 + np.log(r ** 2))
    # reset singularities to 0
    dUdr[mask] = 0
    return dUdr


class MultipleTPS(MultipleAlignment):
    def __init__(self, sources, **kwargs):
        super(MultipleTPS, self).__init__(sources, **kwargs)
        self.tps = [TPS(source, self.target) for source in self.sources]

    @property
    def transforms(self):
        return [tps.transform for tps in self.tps]
