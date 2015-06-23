from functools import partial
from .features import igo, hog

double_igo = partial(igo, double_angles=True)
double_igo.__name__ = 'double_igo'
double_igo.__doc__ = igo.__doc__

sparse_hog = partial(hog, mode='sparse')
sparse_hog.__name__ = 'sparse_hog'
sparse_hog.__doc__ = hog.__doc__

try:
    from .vlfeat import dsift
    fast_dsift = partial(dsift, fast=True, cell_size_vertical=5,
                         cell_size_horizontal=5, num_bins_horizontal=1,
                         num_bins_vertical=1, num_or_bins=8)
    fast_dsift.__name__ = 'fast_dsift'
    fast_dsift.__doc__ = dsift.__doc__
except ImportError:
    pass
