from .features import (gradient, hog, lbp, es, igo, no_op, gaussian_filter,
                       daisy, features_selection_widget)
# Optional dependencies may return nothing.
from .optional import *

from .predefined import sparse_hog, double_igo

from .base import ndfeature, imgfeature
from .visualize import glyph, sum_channels
