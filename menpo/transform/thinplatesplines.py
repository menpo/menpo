import numpy as np
from .base import Transform, Alignment, Invertible
from .rbf import R2LogR2RBF


# Note we inherit from Alignment first to get it's n_dims behavior
class ThinPlateSplines(Alignment, Transform, Invertible):
    r"""
    The thin plate splines (TPS) alignment between 2D `source` and `target`
    landmarks.

    ``kernel`` can be used to specify an alternative kernel function. If
    ``None`` is supplied, the :class:`R2LogR2RBF` kernel will be used.

    Parameters
    ----------
    source : ``(N, 2)`` `ndarray`
        The source points to apply the tps from
    target : ``(N, 2)`` `ndarray`
        The target points to apply the tps to
    kernel : :class:`menpo.transform.rbf.RadialBasisFunction`, optional
        The kernel to apply.
    min_singular_val : `float`, optional
        If the target has points that are nearly coincident, the coefficients
        matrix is rank deficient, and therefore not invertible. Therefore, we
        only take the inverse on the full-rank matrix and drop any singular
        values that are less than this value (close to zero).

    Raises
    ------
    ValueError
        TPS is only with on 2-dimensional data
    """
    def __init__(self, source, target, kernel=None, min_singular_val=1e-4):
        Alignment.__init__(self, source, target)
        if self.n_dims != 2:
            raise ValueError('TPS can only be used on 2D data.')
        if kernel is None:
            kernel = R2LogR2RBF(source.points)
        self.min_singular_val = min_singular_val
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

        # If two points are coincident, or very close to being so, then the
        # matrix is rank deficient and thus not-invertible. Therefore,
        # only take the inverse on the full-rank set of indices.
        _u, _s, _v = np.linalg.svd(self.l)
        keep = _s.shape[0] - sum(_s < self.min_singular_val)
        inv_l = _u[:, :keep].dot(1.0 / _s[:keep, None] * _v[:keep, :])
        self.coefficients = inv_l.dot(self.y.T)

    def _sync_state_from_target(self):
        # now the target is updated, we only have to rebuild the
        # coefficients.
        self._build_coefficients()

    def _apply(self, points, **kwargs):
        r"""
        Performs a TPS transform on the given points.

        Parameters
        ----------
        points : ``(N, D)`` `ndarray`
            The points to transform.

        Returns
        -------
        f : ``(N, D)`` `ndarray`
            The transformed points

        Raises
        ------
        ValueError
            TPS can only be applied to 2D data.
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
        r"""
        :type: ``False``
        """
        return False

    def pseudoinverse(self):
        r"""
        The pseudoinverse of the transform - that is, the transform that
        results from swapping `source` and `target`, or more formally, negating
        the transforms parameters. If the transform has a true inverse this
        is returned instead.

        :type: ``type(self)``
        """
        return ThinPlateSplines(self.target, self.source, kernel=self.kernel)
