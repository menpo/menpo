from .base import Transform, TransformChain
from .homogeneous import *
from .thinplatesplines import ThinPlateSplines
from .piecewiseaffine import PiecewiseAffine
from .rbf import R2LogR2RBF, R2LogRRBF
from .groupalign.procrustes import GeneralizedProcrustesAnalysis
from .compositions import (scale_about_centre, rotate_ccw_about_centre,
                           shear_about_centre, transform_about_centre)
from .tcoords import image_coords_to_tcoords, tcoords_to_image_coords


class WithDims(Transform):
    r"""
    Slices off select dimensions of a shape.

    Parameters
    ----------
    dims : valid numpy array slice
        The slice that will be used on the dimensionality axis of the shape
        under transform. For example, to go from a 3D shape to a 2D one,
        [0, 1] could be provided or np.array([True, True, False]).
    """
    def __init__(self, dims):
        self.dims = dims

    def _apply(self, x, **kwargs):
        # if self.dims is a single number we will return an array with the
        # spatial dimension missing - always reshape to avoid this case.
        return x[:, self.dims].reshape([x.shape[0], -1]).copy()
