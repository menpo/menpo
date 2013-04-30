import numpy as np
from scipy.spatial import distance
from pybug.align.nonrigid.exceptions import TPSError
from pybug.align.nonrigid.base import MultipleNonRigidAlignment, \
    NonRigidAlignment


class TPS(NonRigidAlignment):

    def __init__(self, source, target):
        """

        :param source:
        :param target:
        :raise:
        """
        Alignment.__init__(self, source, target)
        if self.n_dim != 2:
            raise TPSError('TPS can only be used on 2D data.')
        self.V = self.target.T
        self.Y = np.hstack([self.V, np.zeros([2, 3])])
        pairwise_norms = distance.squareform(distance.pdist(self.source))
        self.K = tps_kernel_function(pairwise_norms)
        self.P = np.concatenate(
            [np.ones([self.n_landmarks, 1]), self.source], axis=1)
        O = np.zeros([3, 3])
        top_L = np.concatenate([self.K, self.P], axis=1)
        bot_L = np.concatenate([np.swapaxes(self.P, 0, 1), O], axis=1)
        self.L = np.concatenate([top_L, bot_L], axis=0)
        # b_matrix is the Bending Energy Matrix
        #L_inv = np.linalg.inv(self.L)
        #L_inv_n = L_inv[:-3, :-3]
        #self.b_energy = np.dot(L_inv_n,
        #                np.dot(self.K,
        #                       L_inv_n))
        # store out the coefficients (weightings) of the various
        # components of the warp
        self.coeff = np.linalg.solve(self.L, self.Y.T)

    def mapping(self, coords, affinefree=False):
        """ TPS transform of input coords (f) and the affine-free
     TPS transform of the input coords (f_afree)
    """
        if coords.shape[1] != self.n_dim:
            raise TPSError('TPS can only be used on 2D data.')
        x = coords[..., 0][:, np.newaxis]
        y = coords[..., 1][:, np.newaxis]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_C = self.coeff[-3]
        c_affine_X = self.coeff[-2]
        c_affine_Y = self.coeff[-1]
        # the affine warp component
        f_affine = c_affine_C + c_affine_X * x + c_affine_Y * y
        # calculate a disance matrix (for L2 Norm) between every source
        # and the target
        dist = distance.cdist(self.source, coords)
        kernel_dist = tps_kernel_function(dist)
        # grab the affine free components of the warp
        c_afree = self.coeff[:-3]
        # the affine free warp component
        f_afree = np.sum(
            c_afree[:, np.newaxis, :] * kernel_dist[..., np.newaxis], axis=0)
        if affinefree:
            return f_affine + f_afree, f_afree
        else:
            return f_affine + f_afree

    def view(self):
        self._view_2d()


class ParallelTPS(MultipleNonRigidAlignment):
    def build_TPS_matrices(self):
        self.V = self.target[..., 0].T
        self.Y = np.hstack([self.V, np.zeros([2, 3])])
        pairwise_norms = np.zeros([self.n_landmarks,
                                   self.n_landmarks, self.n_sources])
        for i in range(self.n_sources):
            pairwise_norms[..., i] = distance.squareform(
                distance.pdist(self.sources[..., i]))
            # K is n x n x s matrix of kernel function U evaluated at each of
        # the pairwise norms
        self.K = tps_kernel_function(pairwise_norms)
        self.P = np.concatenate(
            [np.ones([self.n_landmarks, 1, self.n_sources]), self.sources],
            axis=1)
        O = np.zeros([3, 3, self.n_sources])
        top_L = np.concatenate([self.K, self.P], axis=1)
        bot_L = np.concatenate([np.swapaxes(self.P, 0, 1), O], axis=1)
        self.L = np.concatenate([top_L, bot_L], axis=0)

        # b_matrix is the Bending Energy Matrix
        #L_inv = np.linalg.inv(self.L)
        #L_inv_n = L_inv[:-3, :-3]
        #self.b_energy = np.dot(L_inv_n,
        #                np.dot(self.K,
        #                       L_inv_n))
        # store out the coefficients (weightings) of the various
        # components of the warp
        self.coeff = np.zeros(
            [self.Y.shape[1], self.Y.shape[0], self.n_sources])
        for i in range(self.n_sources):
            self.coeff[..., i] = np.linalg.solve(self.L[..., i], self.Y.T)

    def mapping(self, coords):
        """ TPS transform of input coords (f) and the affine-free
        TPS transform of the input coords (f_afree)
        """
        x = coords[..., 0][:, np.newaxis, np.newaxis]
        y = coords[..., 1][:, np.newaxis, np.newaxis]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_C = self.coeff[-3]
        c_affine_X = self.coeff[-2]
        c_affine_Y = self.coeff[-1]
        # the affine warp component
        f_affine = c_affine_C + c_affine_X * x + c_affine_Y * y
        # calculate a distance matrix (for L2 Norm) between every source
        # and the target
        dist = np.zeros([self.n_landmarks, coords.shape[0], self.n_sources])
        for i in range(self.n_sources):
            dist[..., i] = distance.cdist(self.sources[..., i], coords)
        kernel_dist = tps_kernel_function(dist)
        # grab the affine free components of the warp
        c_afree = self.coeff[:-3]
        # the affine free warp component
        f_afree = np.sum(c_afree[:, np.newaxis, ...] *
                         kernel_dist[..., np.newaxis, :], axis=0)
        return f_affine + f_afree, f_afree

    def indiv_mapping_broken(self, coords, i):
        """ TPS transform of input coords (f) and the affine-free
        TPS transform of the input coords (f_afree)
        """
        x = coords[..., 0][:, np.newaxis]
        y = coords[..., 1][:, np.newaxis]
        # calculate the affine coefficients of the warp
        # (C = Constant component, then X, Y respectively)
        c_affine_C = self.coeff[-3, :, i]
        c_affine_X = self.coeff[-2, :, i]
        c_affine_Y = self.coeff[-1, :, i]
        # the affine warp component
        f_affine = c_affine_C + c_affine_X * x + c_affine_Y * y
        # calculate a disance matrix (for L2 Norm) between every source
        # and the target
        dist = distance.cdist(self.sources[..., i], coords)
        kernel_dist = tps_kernel_function(dist)
        # grab the affine free components of the warp
        c_afree = self.coeff[:-3, :, i]
        # the affine free warp component
        f_afree = np.sum(c_afree[..., np.newaxis] *
                         kernel_dist[:, np.newaxis, :], axis=0)
        return f_affine + f_afree.T, f_afree.T


def tps_kernel_function(r):
    #TPS_KERNEL_FUNCTION Returns a matrix of evaluations of radial distances
    # store all singularity positions
    mask = r == 0
    r[mask] = 1
    U = r ** 2 * (np.log(r ** 2))
    # reset singularities to 0
    U[mask] = 0
    return U


