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
    def __init__(self, dims):
        self.dims = dims

    def _apply(self, x, **kwargs):
        return x[:, self.dims].copy()

    @property
    def n_dims_output(self):
        return len(self.dims)
