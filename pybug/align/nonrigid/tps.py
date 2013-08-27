import numpy as np
from scipy.spatial import distance
from pybug.transform.tps import TPSTransform
from pybug.align.nonrigid.base import NonRigidAlignment
from pybug.exceptions import DimensionalityError
from pybug.align.base import MultipleAlignment
from pybug.basis.rbf import R2LogR2


class TPS(NonRigidAlignment):
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
    DimensionalityError
        TPS is only supported on 2-dimensional data
    """

    def __init__(self, source, target, kernel=None):
        super(TPS, self).__init__(source, target)
        if self.n_dims != 2:
            raise DimensionalityError('TPS can only be used on 2D data.')
        self.V = self.target.T.copy()
        self.Y = np.hstack([self.V, np.zeros([2, 3])])
        pairwise_norms = distance.cdist(self.source, self.source)
        if kernel is None:
            kernel = R2LogR2()
        self.kernel = kernel
        self.K = self.kernel.phi(pairwise_norms)
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
        r"""
        The TPS transform object.

        :type: :class:`pybug.transform.tps.TPSTransform`
        """
        return self._transform_object

    def view(self, image=False):
        r"""
        View the object. This plots the source points and vectors that
        represent the shift from source to target.

        Parameters
        ----------
        image : bool, optional
            If ``True`` the vectors are plotted on top of an image

            Default: ``False``
        """
        self._view_2d(image=image)


class MultipleTPS(MultipleAlignment):
    r"""
    Applies thin plate spline (TPS) alignment to multiple sources.

    Parameters
    ----------
    sources :  (N, 2) list of ndarray
    """

    def __init__(self, sources, **kwargs):
        super(MultipleTPS, self).__init__(sources, **kwargs)
        self.tps = [TPS(source, self.target) for source in self.sources]

    @property
    def transforms(self):
        r"""
        The list of TPS transforms for each source in sources

        :type: list of :class:`pybug.transform.tps.TPSTransform`
        """
        return [tps.transform for tps in self.tps]
