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
        # My matlab code loops over all vertices but maybe in python I can
        # get away without a loop because of the broadcasting...

        # This needs to be coded, I can do that between this eening and
        # tomorrow morning,
        dL_dx = np.zeros(self.tps.L.shape + (self.tps.source.shapes[0],))
        #dL_dx[]

        # TPS kernel (nonlinear + affine) # This bit should be OK
        dist = distance.cdist(self.tps.source, points)
        kernel_dist = r_2_log_r_2_ker,nel(dist)
        k = np.concatenate([kernel_dist, np.ones((1, points.shape[0])),
                            points.T], axis=0)
        inv_L = np.linalg.inv(self.tps.L)

        return self.tps.Y * (-inv_L * dL_dx * inv_L) * k

        pass

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
    Radial basis function for TPS.
    """
    mask = r == 0
    r[mask] = 1
    U = r ** 2 * (np.log(r ** 2))
    # reset singularities to 0
    U[mask] = 0
    return U


class MultipleTPS(MultipleAlignment):
    def __init__(self, sources, **kwargs):
        super(MultipleTPS, self).__init__(sources, **kwargs)
        self.tps = [TPS(source, self.target) for source in self.sources]

    @property
    def transforms(self):
        return [tps.transform for tps in self.tps]
