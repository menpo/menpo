import numpy as np
from scipy.spatial import distance
from pybug.transform.tps import TPSTransform
from pybug.align.nonrigid.base import NonRigidAlignment
from pybug.exceptions import DimensionalityError
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
        self.kernel = kernel
        # TODO: kernels need some sort of structured form
        self.kernel_derivative = r_2_log_r_2_kernel_derivative
        self.K = self.kernel(pairwise_norms)
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

    def view(self, image=False):
        self._view_2d(image=image)


# TODO: kernels need some sort of structured form
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


# TODO: kernels need some sort of structured form
def r_2_log_r_2_kernel_derivative(r):
    """
    Derivative of the radial basis function for TPS.
    """
    mask = r == 0
    r[mask] = 1
    dUdr = 2 * (1 + np.log(r ** 2))
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
