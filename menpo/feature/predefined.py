from functools import partial
from .features import hog, igo

sparse_hog = partial(hog, mode='sparse')
double_igo = partial(igo, double_angles=True)
