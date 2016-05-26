from menpo.base import partial_doc
from .features import igo, hog

double_igo = partial_doc(igo, double_angles=True)
sparse_hog = partial_doc(hog, mode='sparse')
