from menpo.base import partial_doc
from .features import igo, hog

double_igo = partial_doc(igo, double_angles=True, menpo_f_name='double_igo')
sparse_hog = partial_doc(hog, mode='sparse', menpo_f_name='sparse_hog')
