import numpy as np
from .base import Transform, Alignment, Invertible
from .rbf import R2LogR2RBF


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

    def pseudoinverse(self):
        return ThinPlateSplines(self.target, self.source, kernel=self.kernel)
