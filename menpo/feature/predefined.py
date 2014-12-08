from functools import partial
from .features import hog, igo, dsift

sparse_hog = partial(hog, mode='sparse')
double_igo = partial(igo, double_angles=True)
sparse_hog.__name__ = 'sparse_hog'
double_igo.__name__ = 'double_igo'

fast_dsift = partial(dsift, fast=True, window_size=5, geometry=(1, 1, 8))
fast_dsift.__name__ = 'fast_dsift'
