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
        The TPS alignmnet between 2D source and target landmarks. kernel can
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
        self.K = kernel(pairwise_norms).T
        self.P = np.concatenate(
            [np.ones([self.n_landmarks, 1]), self.source], axis=1)
        O = np.zeros([3, 3])
        top_L = np.concatenate([self.K, self.P], axis=1)
        bot_L = np.concatenate([np.swapaxes(self.P, 0, 1), O], axis=1)
        self.L = np.concatenate([top_L, bot_L], axis=0)
        self.coeff = np.linalg.solve(self.L, self.Y.T)
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

    def _apply(self, x, affinefree=False):
        """ TPS transform of input x (f) and the affine-free
        TPS transform of the input x (f_afree)
        """
        if x.shape[1] != self.n_dim:
            raise TPSError('TPS can only be used on 2D data.')
        x = x[..., 0][:, np.newaxis]
        y = x[..., 1][:, np.newaxis]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_C = self.tps.coeff[-3]
        c_affine_X = self.tps.coeff[-2]
        c_affine_Y = self.tps.coeff[-1]
        # the affine warp component
        f_affine = c_affine_C + c_affine_X * x + c_affine_Y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        dist = distance.cdist(self.tps.source, x)
        kernel_dist = r_2_log_r_2_kernel(dist)
        # grab the affine free components of the warp
        c_afree = self.tps.coeff[:-3]
        # the affine free warp component
        f_afree = np.sum(
            c_afree[:, np.newaxis, :] * kernel_dist[..., np.newaxis], axis=0)
        if affinefree:
            return f_affine + f_afree, f_afree
        else:
            return f_affine + f_afree


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
