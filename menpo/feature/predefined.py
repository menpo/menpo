from functools import partial
from .features import igo, hog

double_igo = partial(igo, double_angles=True)
double_igo.__name__ = 'double_igo'
double_igo.__doc__ = igo.__doc__

sparse_hog = partial(hog, mode='sparse')
sparse_hog.__name__ = 'sparse_hog'
sparse_hog.__doc__ = hog.__doc__
