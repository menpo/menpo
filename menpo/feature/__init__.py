from .features import (gradient, hog, lbp, es, igo, no_op, gaussian_filter,
                       daisy, features_selection_widget)
# If cyvlfeat is not installed, then access to vlfeat features should be blocked
try:
    from .vlfeat import dsift
except ImportError:
    pass

from .predefined import sparse_hog, double_igo
try:
    from .predefined import fast_dsift
except ImportError:
    pass

from .base import ndfeature, imgfeature
from .visualize import glyph, sum_channels
