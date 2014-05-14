import numpy as np

from menpo.base import DX, DL

from .base import Transform, Alignment, Invertible
from .rbf import R2LogR2RBF


# Note we inherit from Alignment first to get it's n_dims behavior
class ThinPlateSplines(Alignment, Transform, Invertible, DX, DL):
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
            kernel = R2LogR2RBF(source.points)
        self.kernel = kernel
        # k[i, j] is the rbf weighting between source i and j
        # (of course, k is thus symmetrical and it's diagonal nil)
        self.k = self.kernel.apply(self.source.points)
        # p is a homogeneous version of the source points
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

    def d_dx(self, points):
        r"""
        The first order derivative of this TPS warp wrt spatial changes
        evaluated at points.

        Parameters
        ----------
        points: ndarray shape (n_points, n_dims)
            The spatial points at which the derivative should be evaluated.

        Returns
        -------
        d_dx: ndarray shape (n_points, n_dims, n_dims)
            The jacobian wrt spatial changes.

            d_dx[i, j, k] is the scalar differential change that the
            j'th dimension of the i'th point experiences due to a first order
            change in the k'th dimension.

        """
        dk_dx = np.zeros((points.shape[0] + 3,   # i
                          self.source.n_points,  # k
                          self.source.n_dims))   # l
        dk_dx[:-3, :] = self.kernel.d_dl(points)

        affine_derivative = np.array([[0, 0],
                                      [1, 0],
                                      [0, 1]])
        dk_dx[-3:, :] = affine_derivative[:, None]

        return np.einsum('ij, ikl -> klj', self.coefficients, dk_dx)

    def d_dl(self, points):
        """
        Calculates the Jacobian of the TPS warp wrt to the source landmarks
        assuming that he target is equal to the source. This is a special
        case of the Jacobian wrt to the source landmarks that is used in AAMs
        to weight the relative importance of each pixel in the reference
        frame wrt to each one of the source landmarks.

        dW_dl =      dOmega_dl         *  k(points)
              = T *     d_L**-1_dl     *  k(points)
              = T * -L**-1 dL_dl L**-1 *  k(points)

        # per point
        (c, d) = (d, c+3) (c+3, c+3) (c+3, c+3, c, d) (c+3, c+3) (c+3)
        (c, d) = (d,            c+3) (c+3, c+3, c, d) (c+3,)
        (c, d) = (d,               ) (          c, d)
        (c, d) = (                 ) (          c, d)

        Parameters
        ----------
        points : (n_points, n_dims)
            Points at which the Jacobian will be evaluated.

        Returns
        -------
        dW/dl : (n_points, n_params, n_dims) ndarray
            The Jacobian of the transform wrt to the source landmarks evaluated
            at the previous points and assuming that the target is equal to
            the source.
        """
        n_centres = self.n_points
        n_points = points.shape[0]

        # TPS kernel (nonlinear + affine)

        # for each input, evaluate the rbf
        # (n_points, n_centres)
        k_points = self.kernel.apply(points)

        # k_points with (1, x, y) appended to each point
        # (n_points, n_centres+3)  - 3 is (1, x, y) for affine component
        k = np.hstack([k_points, np.ones([n_points, 1]), points])

        # (n_centres+3, n_centres+3)
        inv_L = np.linalg.inv(self.l)

        # Taking the derivative of L for changes in l must yield an x,y change
        # for each centre.
        # (n_centres+3, n_centres+3, n_centres, n_dims)
        dL_dl = np.zeros(self.l.shape + (n_centres, 2))

        # take the derivative of the kernel wrt centres at the centres
        # SHOULD be (n_centres, n_dims, n_centres, n_dims)
        # IS        (n_centres,         n_centres, n_dims
        dK_dl_at_tgt = self.kernel.d_dl(self.source.points)

        # we want to build a tensor where for each slice where
        # dK_dl[i, j, k, l] is the derivative wrt the l'th dimension of the
        # i'th centre for L[j, k] -> first axis is just looping over centres
        # and last looping over dims
        # (n_centres, n_centres, n_centres, n_dims)
        dK_dl = np.zeros((n_centres, ) + dK_dl_at_tgt.shape)

        # make a linear iterator over the centres
        iter = np.arange(n_centres)

        # efficiently build the repeated pattern for dK_dl
        # note that the repetition over centres happens over axis 0
        # and the dims axis is the last
        # so dK_dl[0, ..., 0] corresponds to dK/dx0 in Joan's paper
        #    dK_dl[3, ..., 1] corresponds to dK_dy3 in Joan's paper
        dK_dl[iter, iter] = dK_dl_at_tgt[iter]
        dK_dl[iter, :, iter] = dK_dl_at_tgt[:, iter]

        # prepare memory for the answer
        # SHOULD be (n_points, n_dims, n_centres, n_dims)
        # IS        (n_points,       , n_centres, n_dims)
        dW_dl = np.zeros((n_points, n_centres, 2))

        # pretend the target is equal to the source
        # (n_dims, n_centres+3)
        pseudo_target = np.hstack([self.source.points.T, np.zeros([2, 3])])

        for i in np.arange(n_centres):
            # dP_dli (n_centres, n_points, n_dims, n_dims)
            dP_dli = np.zeros(self.p.shape + (2,))
            dP_dli[i, 1, 0] = -1
            dP_dli[i, 2, 1] = -1

            dL_dl[:n_centres, :n_centres, i] = dK_dl[i]
            dL_dl[:n_centres, n_centres:, i] = dP_dli
            dL_dl[n_centres:, :n_centres, i] = np.swapaxes(dP_dli, 0, 1)

            omega_x = -inv_L.dot(dL_dl[..., i, 0].dot(inv_L))
            omega_y = -inv_L.dot(dL_dl[..., i, 1].dot(inv_L))
            dW_dl[:, i, 0] = k.dot(omega_x).dot(pseudo_target[0])
            dW_dl[:, i, 1] = k.dot(omega_y).dot(pseudo_target[1])

        return dW_dl
