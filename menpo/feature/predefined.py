from functools import partial
from .features import hog, igo

sparse_hog = partial(hog, mode='sparse')
double_igo = partial(igo, double_angles=True)
sparse_hog.__name__ = 'sparse_hog'
sparse_hog.__doc__ = hog.__doc__
double_igo.__name__ = 'double_igo'
double_igo.__doc__ = igo.__doc__
